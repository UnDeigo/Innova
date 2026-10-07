#!/usr/bin/env python3
"""
Prototipo con VIDEO: detecta cuando un vehículo se queda DETENIDO dentro de la ciclovía.

Usa la misma calibración en metros que deteccion_imagen.py (ancho y largo de un tramo
de ciclovía + 4 clics), así que "detenido" se mide en metros reales, no en píxeles.

Cómo decide:
  1. YOLO + ByteTrack detectan y siguen cada vehículo.
  2. Un vehículo cuenta como "en la ciclovía" si al menos --solape de su borde inferior
     (por defecto 25%) cae dentro de la franja de la ciclovía.
  3. Si pasa más de --segundos (por defecto 10) dentro de la ciclovía SIN moverse más de
     --mov-m metros (por defecto 0,8), se registra una invasión.
  4. Por cada evento guarda foto, recorte del vehículo y clip, y genera eventos.csv e informe.html.

Ejemplos:
  python deteccion_video.py video.mp4 --calibrar --ancho 2.0 --largo 6.0 --ver
  python deteccion_video.py video.mp4 --segundos 10 --inicio "2026-10-10 08:30"
  python deteccion_video.py --evaluar salida_video/video --perdidos 1   (carpeta de ESE video)
"""
import argparse
import bisect
import csv
import glob
import html
import os
import sys
from datetime import datetime, timedelta

import cv2
import numpy as np

import deteccion_imagen as di


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def parsear_inicio(texto):
    if not texto:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(texto, fmt)
        except ValueError:
            pass
    sys.exit('Formato de --inicio inválido. Usa "2026-10-10 08:30"')


def fmt_t(t, inicio_dt):
    if inicio_dt is not None:
        return (inicio_dt + timedelta(seconds=t)).strftime("%d-%m-%Y %H:%M:%S")
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def abrir_writer(ruta, fps, tam):
    for codec in ("mp4v", "avc1"):  # mp4v primero: funciona en cualquier PC sin DLL extra
        vw = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*codec), fps, tam)
        if vw.isOpened():
            return vw
    sys.exit("No se pudo crear el clip de video (codec no disponible).")


def dibujar_franja(img, Hinv, ancho, largo):
    """Pinta la franja de la ciclovía (calibrada + extensión) sobre la imagen, in situ."""
    e1, e2 = di.extension_dibujable(Hinv, ancho, largo)
    franja = di.transformar(Hinv, [[0, -e1], [ancho, -e1],
                                   [ancho, largo + e2], [0, largo + e2]]).astype(np.int32)
    capa = img.copy()
    cv2.fillPoly(capa, [franja], (255, 200, 0))
    img[:] = cv2.addWeighted(capa, 0.25, img, 0.75, 0)
    cv2.polylines(img, [franja], True, (255, 200, 0), 2)


# ----------------------------------------------------------------------------
# Compensación del movimiento de la cámara (si se grabó a mano)
# ----------------------------------------------------------------------------
class Estabilizador:
    """Estima cuánto se movió la cámara respecto al PRIMER cuadro (donde se calibró),
    para que la ciclovía quede pegada al suelo aunque el video tiemble o se desplace."""

    def __init__(self, ref_bgr, ignorar_izq=0.0):
        g = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        self.centro = np.float32([[[w / 2.0, h / 2.0]]])
        self.mask = np.full((h, w), 255, np.uint8)
        self.mask[:, :int(w * ignorar_izq)] = 0   # zona a ignorar (ej. una baranda borrosa)
        self.orb = cv2.ORB_create(3000)
        self.kp0, self.des0 = self.orb.detectAndCompute(g, self.mask)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.G = np.eye(3)
        self.fallos = 0
        self.max_desp = 0.0

    def a_referencia(self, frame):
        """Homografía que lleva los píxeles de este cuadro a los del cuadro de referencia."""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(g, self.mask)
        if des is None or self.des0 is None:
            self.fallos += 1
            return self.G
        m = self.bf.match(des, self.des0)
        if len(m) < 15:
            self.fallos += 1
            return self.G
        src = np.float32([kp[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
        dst = np.float32([self.kp0[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
        G, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if G is None or int(inl.sum()) < 12:
            self.fallos += 1
            return self.G
        self.G = G
        p = cv2.perspectiveTransform(self.centro, G)[0][0]
        self.max_desp = max(self.max_desp, float(np.linalg.norm(p - self.centro[0][0])))
        return G


def solape_cajas(a, b):
    """Qué tanto se superponen dos cajas (x1, y1, x2, y2): el mayor entre IoU y
    intersección / área de la caja más chica (cubre cajas duplicadas una dentro de otra)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    if inter <= 0 or area_a <= 0 or area_b <= 0:
        return 0.0
    return max(inter / (area_a + area_b - inter), inter / min(area_a, area_b) * 0.9)


# ----------------------------------------------------------------------------
# Detector con seguimiento
# ----------------------------------------------------------------------------
def crear_tracker(modelo_nombre, conf, imgsz):
    from ultralytics import YOLO

    modelo = YOLO(modelo_nombre)

    def detectar(frame):
        """Devuelve lista de (id, clase, x1, y1, x2, y2)."""
        r = modelo.track(frame, persist=True, conf=conf, imgsz=imgsz,
                         classes=list(di.CLASES_VEHICULO), tracker="bytetrack.yaml",
                         verbose=False)[0]
        salida = []
        if r.boxes is None or r.boxes.id is None:
            return salida
        for caja, tid, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist(),
                                r.boxes.cls.int().cpu().tolist()):
            x1, y1, x2, y2 = (float(v) for v in caja)
            salida.append((tid, di.CLASES_VEHICULO.get(c, str(c)), x1, y1, x2, y2))
        return salida

    return detectar


# ----------------------------------------------------------------------------
# Lógica principal: ¿hay un vehículo detenido dentro de la ciclovía?
# ----------------------------------------------------------------------------
def procesar_video(args, H, Hinv, ancho, largo, detectar, cap, fps, carpeta, inicio_dt,
                   estab=None, homs=None, stats=None):
    estados, eventos = {}, []
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.saltar != 0:
            continue
        t = idx / fps
        if estab is not None:
            Ht = H @ estab.a_referencia(frame)   # píxeles de ESTE cuadro -> metros
            Hinv_t = np.linalg.inv(Ht)
        else:
            Ht, Hinv_t = H, Hinv
        if homs is not None:
            homs[idx] = Hinv_t
        vista = frame.copy() if args.ver else None
        if vista is not None:
            dibujar_franja(vista, Hinv_t, ancho, largo)

        for tid, clase, x1, y1, x2, y2 in detectar(frame):
            mundo = di.transformar(Ht, di.borde_inferior(x1, x2, y2))
            dentro = ((mundo[:, 0] >= 0) & (mundo[:, 0] <= ancho) &
                      (mundo[:, 1] >= -di.EXTENSION_M) & (mundo[:, 1] <= largo + di.EXTENSION_M))
            pct = float(dentro.mean())
            en_ciclovia = pct >= args.solape
            if stats is not None:
                r = stats.setdefault(tid, {"clase": clase, "n": 0, "t0": t, "t1": t, "pct_sum": 0.0,
                                           "pct_max": 0.0, "parado_max": 0.0, "evento": False})
                r["n"] += 1
                r["t1"] = t
                r["pct_sum"] += pct
                r["pct_max"] = max(r["pct_max"], pct)
            pos = mundo[di.MUESTRAS // 2]  # centro del borde inferior, en metros
            st = estados.get(tid)
            color, etiqueta = (0, 200, 0), f"{clase} #{tid}"

            if en_ciclovia:
                reiniciar = (st is None or (t - st["ultimo"]) > args.tolerancia or
                             float(np.linalg.norm(pos - st["pos"])) > args.mov_m)
                if reiniciar:
                    st = {"inicio": t, "pos": pos, "ultimo": t, "evento": None,
                          "cajas": {}, "pct": []}
                    estados[tid] = st
                else:
                    st["ultimo"] = t
                st["cajas"].setdefault(idx, (x1, y1, x2, y2))  # no pisar la caja de otro track fusionado
                st["pct"].append(pct)
                parado = t - st["inicio"]
                if stats is not None:
                    stats[tid]["parado_max"] = max(stats[tid]["parado_max"], parado)

                prev = None
                if st["evento"] is None and parado >= min(args.reenganche, args.segundos):
                    # ¿Es el mismo vehículo de un evento anterior? (misma zona, poco tiempo de diferencia).
                    # Si ya estaba marcado, basta con verlo quieto unos segundos para continuar ese evento.
                    # Pasa cuando el rastreador le cambia el número o detecta dos cajas del mismo vehículo.
                    # OJO: además de cercanía en tiempo y espacio se exige que la caja SE SOLAPE con la
                    # última caja del evento. Si no, un vehículo DISTINTO detenido a pocos metros de otro
                    # ya marcado quedaba absorbido por ese evento y nunca generaba foto ni clip propios.
                    for e in reversed(eventos):
                        if (st["inicio"] - e["fin"] <= args.fusionar and
                                float(np.linalg.norm(pos - e["pos"])) <= args.fusionar_m and
                                solape_cajas((x1, y1, x2, y2), e["caja"]) >= args.fusionar_iou):
                            prev = e
                            break
                if prev is not None:
                    prev["cajas"].update(st["cajas"])
                    st["cajas"] = prev["cajas"]
                    prev["pct"].extend(st["pct"])
                    st["pct"] = prev["pct"]
                    prev["fin"] = max(prev["fin"], t)
                    prev["pos"] = pos
                    prev["caja"] = (x1, y1, x2, y2)
                    st["evento"] = prev
                    if stats is not None:
                        stats[tid]["fusion"] = prev["n"]
                    print(f"  (id #{tid} unido al evento {prev['n']} en {fmt_t(t, inicio_dt)}: "
                          f"misma zona y cajas solapadas)")
                elif st["evento"] is None and parado >= args.segundos:
                    n = len(eventos) + 1
                    foto, recorte = f"evento_{n:03d}_foto.jpg", f"evento_{n:03d}_vehiculo.jpg"
                    img = frame.copy()
                    dibujar_franja(img, Hinv_t, ancho, largo)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    di.texto(img, f"{clase} #{tid} detenido en ciclovia - {fmt_t(t, inicio_dt)}",
                             (int(x1), max(int(y1) - 10, 25)), (0, 0, 255),
                             max(0.5, frame.shape[1] / 1600.0))
                    cv2.imwrite(os.path.join(carpeta, foto), img)
                    h, w = frame.shape[:2]
                    mx, my = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
                    cv2.imwrite(os.path.join(carpeta, recorte),
                                frame[max(int(y1) - my, 0):min(int(y2) + my, h),
                                      max(int(x1) - mx, 0):min(int(x2) + mx, w)])
                    ev = {"n": n, "id": tid, "clase": clase, "inicio": st["inicio"], "fin": t,
                          "foto": foto, "recorte": recorte, "clip": "",
                          "cajas": st["cajas"], "pct": st["pct"], "pos": pos,
                          "caja": (x1, y1, x2, y2),
                          "f_foto": idx, "caja_foto": (x1, y1, x2, y2), "Hinv_foto": Hinv_t}
                    eventos.append(ev)
                    st["evento"] = ev
                    print(f"  Evento {n}: {clase} #{tid} detenido en la ciclovía desde "
                          f"{fmt_t(st['inicio'], inicio_dt)}")
                elif st["evento"] is not None:
                    st["evento"]["fin"] = max(st["evento"]["fin"], t)
                    st["evento"]["pos"] = pos
                    st["evento"]["caja"] = (x1, y1, x2, y2)

                if st["evento"] is not None and stats is not None:
                    stats[tid]["evento"] = True
                if st["evento"] is not None:
                    color, etiqueta = (0, 0, 255), f"{clase} #{tid} DETENIDO {parado:.0f}s"
                else:
                    color, etiqueta = (0, 220, 255), f"{clase} #{tid} en ciclovia {parado:.0f}s"

            if vista is not None:
                cv2.rectangle(vista, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(vista, etiqueta, (int(x1), max(int(y1) - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if vista is not None:
            cv2.putText(vista, f"t = {fmt_t(t, inicio_dt)}   eventos: {len(eventos)}   (q: salir)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Prototipo video", vista)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if idx % 300 == 0:
            print(f"  ... procesado {fmt_t(t, inicio_dt)}")

    if args.ver:
        cv2.destroyAllWindows()
    return eventos


def imprimir_diagnostico(stats, args, inicio_dt=None):
    """Tabla con TODOS los vehículos que el modelo vio y por qué no se marcó cada uno."""
    print("\n--- Diagnóstico: vehículos que el modelo vio en el video ---")
    print(f"{'id':>3}  {'clase':<7} {'visto':<13} {'% en ciclovía':<14} {'máx. quieto':<12} resultado")
    for tid, e in sorted(stats.items()):
        prom = e["pct_sum"] / e["n"]
        visto = f"{fmt_t(e['t0'], inicio_dt)}-{fmt_t(e['t1'], inicio_dt)}"
        if e["evento"]:
            res = f"EVENTO (unido al evento {e['fusion']})" if e.get("fusion") else "EVENTO"
        elif e["t1"] - e["t0"] < 1.5:
            res = "visto muy poco (probablemente ruido o pasó de largo)"
        elif e["pct_max"] < args.solape:
            res = (f"FUERA de la zona (máx. {e['pct_max']:.0%}, se pide {args.solape:.0%}): "
                   "revisa la zona o baja --solape")
        elif e["parado_max"] < args.segundos:
            res = f"en la zona, pero quieto solo {e['parado_max']:.0f} s (se piden {args.segundos:g} s)"
        else:
            res = "revisar"
        print(f"{tid:>3}  {e['clase']:<7} {visto:<13} {prom:>5.0%} / {e['pct_max']:<5.0%}  "
              f"{e['parado_max']:>5.0f} s      {res}")
    print("Si un vehículo que ves en el video NO aparece en esta tabla, el modelo no lo detectó: "
          "prueba --conf 0.2 o --modelo yolov8m.pt")


# ----------------------------------------------------------------------------
# Clips, CSV e informe
# ----------------------------------------------------------------------------
def cortar_clips(video, eventos, Hinv, ancho, largo, carpeta, fps, tam, inicio_dt, saltar, homs=None):
    cap = cv2.VideoCapture(video)
    claves_h = sorted(homs) if homs else []
    for ev in eventos:
        t0 = max(0.0, ev["inicio"] - 3)
        t1 = min(ev["fin"] + 3, ev["inicio"] + 40)  # máx. ~40 s por clip
        f0 = int(t0 * fps)
        nombre = f"evento_{ev['n']:03d}_clip.mp4"
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        vw = abrir_writer(os.path.join(carpeta, nombre), fps, tam)
        claves = sorted(ev["cajas"])
        for k in range(int((t1 - t0) * fps)):
            ok, fr = cap.read()
            if not ok:
                break
            f = f0 + k
            Hi = Hinv
            if claves_h:
                jh = bisect.bisect_right(claves_h, f) - 1
                Hi = homs[claves_h[max(jh, 0)]]
            dibujar_franja(fr, Hi, ancho, largo)
            j = bisect.bisect_right(claves, f) - 1
            if j >= 0 and f - claves[j] <= saltar * 2:
                x1, y1, x2, y2 = ev["cajas"][claves[j]]
                cv2.rectangle(fr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(fr, fmt_t(f / fps, inicio_dt), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            vw.write(fr)
        vw.release()
        ev["clip"] = nombre
    cap.release()


def regenerar_fotos(video, eventos, ancho, largo, carpeta, inicio_dt):
    """Rehace la foto de cada evento al terminar el análisis, cuando ya se conoce la duración
    total: la etiqueta muestra inicio - fin (duración) en vez de la hora en que se cruzó el umbral."""
    cap = cv2.VideoCapture(video)
    for ev in eventos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ev["f_foto"])
        ok, fr = cap.read()
        if not ok:
            continue  # se conserva la foto original
        dibujar_franja(fr, ev["Hinv_foto"], ancho, largo)
        x1, y1, x2, y2 = ev["caja_foto"]
        cv2.rectangle(fr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        dur = ev["fin"] - ev["inicio"]
        lineas = [f"{ev['clase']} #{ev['id']} detenido en ciclovia ({dur:.0f} s)",
                  f"{fmt_t(ev['inicio'], inicio_dt)} - {fmt_t(ev['fin'], inicio_dt)}"]
        h, w = fr.shape[:2]
        escala = max(0.5, w / 1600.0)
        fuente = cv2.FONT_HERSHEY_SIMPLEX
        while escala > 0.3 and max(cv2.getTextSize(l, fuente, escala, 2)[0][0] for l in lineas) > w - 12:
            escala -= 0.04  # que el texto quepa en imágenes angostas (videos verticales)
        ancho_txt = max(cv2.getTextSize(l, fuente, escala, 2)[0][0] for l in lineas)
        alto_lin = cv2.getTextSize("Ag", fuente, escala, 2)[0][1] + 10
        tx = int(max(6, min(x1, w - ancho_txt - 6)))
        ty = int(y1) - 10 - alto_lin * (len(lineas) - 1)
        ty = max(ty, alto_lin)  # si la caja está pegada arriba, el texto baja pero no se sale
        for k, l in enumerate(lineas):
            di.texto(fr, l, (tx, ty + k * alto_lin), (0, 0, 255), escala)
        cv2.imwrite(os.path.join(carpeta, ev["foto"]), fr)
    cap.release()


def escribir_csv(eventos, carpeta, inicio_dt):
    ruta = os.path.join(carpeta, "eventos.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["evento", "clase", "id_tracker", "inicio", "fin", "duracion_s",
                    "fraccion_en_ciclovia", "foto", "recorte", "clip", "correcto (si/no)"])
        for ev in eventos:
            w.writerow([ev["n"], ev["clase"], ev["id"], fmt_t(ev["inicio"], inicio_dt),
                        fmt_t(ev["fin"], inicio_dt), round(ev["fin"] - ev["inicio"], 1), f'{np.mean(ev["pct"]):.2f}',
                        ev["foto"], ev["recorte"], ev["clip"], ""])


def escribir_informe(eventos, carpeta, inicio_dt, args):
    por_minuto = {}
    for ev in eventos:
        clave = fmt_t(ev["inicio"] - (ev["inicio"] % 60), inicio_dt)
        por_minuto[clave] = por_minuto.get(clave, 0) + 1
    maximo = max(por_minuto.values()) if por_minuto else 1
    barras = "".join(
        f'<div class="fila"><span>{html.escape(k)}</span>'
        f'<div class="barra" style="width:{int(180 * v / maximo)}px"></div> {v}</div>'
        for k, v in sorted(por_minuto.items()))
    filas = "".join(
        f'<tr><td>{ev["n"]}</td>'
        f'<td><a href="{ev["foto"]}"><img src="{ev["foto"]}" width="240"></a></td>'
        f'<td><a href="{ev["recorte"]}"><img src="{ev["recorte"]}" height="90"></a></td>'
        f'<td>{html.escape(ev["clase"])}</td><td>{html.escape(fmt_t(ev["inicio"], inicio_dt))} &ndash; {html.escape(fmt_t(ev["fin"], inicio_dt))}</td>'
        f'<td>{ev["fin"] - ev["inicio"]:.0f} s</td><td>{np.mean(ev["pct"]):.0%}</td>'
        f'<td><a href="{ev["clip"]}">Ver clip</a></td></tr>' for ev in eventos
    ) or '<tr><td colspan="8">No se detectaron vehículos detenidos en la ciclovía.</td></tr>'
    pagina = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>Informe de invasiones de ciclovía</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:24px;color:#1c1c1c}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{border:1px solid #ccc;padding:8px;text-align:left;vertical-align:middle}}
 th{{background:#f0f0f0}}
 .fila{{display:flex;align-items:center;gap:8px;margin:2px 0;font-size:14px}}
 .fila span{{width:150px}} .barra{{height:14px;background:#d9534f}}
 .aviso{{background:#fff7e0;border:1px solid #e6c200;padding:10px;margin:12px 0}}
</style></head><body>
<h1>Vehículos detenidos en la ciclovía</h1>
<p>Video: {html.escape(os.path.basename(args.video))} &middot; Criterio: más de {args.segundos:g} s
dentro de la ciclovía, moviéndose menos de {args.mov_m:g} m &middot; Eventos: <b>{len(eventos)}</b></p>
<div class="aviso">Prototipo. Cada evento debe ser <b>validado por una persona</b> antes de
cualquier acción. La patente debe leerse en el recorte del vehículo.</div>
<h2>Eventos por minuto</h2>{barras or "<p>Sin eventos.</p>"}
<h2>Detalle</h2>
<table><tr><th>#</th><th>Foto</th><th>Vehículo</th><th>Clase</th><th>Inicio - Fin</th>
<th>Duración</th><th>% en ciclovía</th><th>Clip</th></tr>{filas}</table>
</body></html>"""
    with open(os.path.join(carpeta, "informe.html"), "w", encoding="utf-8") as f:
        f.write(pagina)


def evaluar(carpeta, perdidos):
    si = no = 0
    with open(os.path.join(carpeta, "eventos.csv"), encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            v = di.si_no(fila.get("correcto (si/no)"))
            if v is True:
                si += 1
            elif v is False:
                no += 1
    total = si + no
    if total == 0:
        sys.exit("Completa la columna 'correcto (si/no)' de eventos.csv antes de evaluar.")
    print(f"Eventos validados: {total}  (correctos: {si}, falsos positivos: {no})")
    print(f"Precisión: {si / total:.0%}  (de lo que detectó, cuánto era real)")
    if perdidos is not None:
        print(f"Recall:    {si / (si + perdidos):.0%}  (no detectó {perdidos} invasión(es) real(es))")
    else:
        print("Para el recall agrega --perdidos N (invasiones reales que se le escaparon).")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Vehículos detenidos en ciclovía (video)")
    ap.add_argument("video", nargs="?", help="archivo de video (cámara fija)")
    ap.add_argument("--calibrar", action="store_true", help="(re)hacer la calibración")
    ap.add_argument("--ancho", type=float, help="ancho real del tramo de ciclovía en metros")
    ap.add_argument("--largo", type=float, help="largo real del tramo calibrado en metros")
    ap.add_argument("--puntos", help='4 puntos "x,y;x,y;x,y;x,y" (cerca-izq, cerca-der, lejos-der, lejos-izq)')
    ap.add_argument("--segundos", type=float, default=10, help="tiempo detenido para marcar invasión")
    ap.add_argument("--mov-m", dest="mov_m", type=float, default=0.8,
                    help="movimiento máximo en metros para considerarlo detenido")
    ap.add_argument("--tolerancia", type=float, default=3.0, help="segundos sin detectar antes de reiniciar")
    ap.add_argument("--fusionar", type=float, default=15.0,
                    help="unir eventos del mismo lugar separados por menos de N segundos")
    ap.add_argument("--fusionar-m", dest="fusionar_m", type=float, default=2.0,
                    help="...si están a menos de N metros uno del otro")
    ap.add_argument("--fusionar-iou", dest="fusionar_iou", type=float, default=0.3,
                    help="...y solo si su caja se solapa al menos esta fracción con la del evento (0 = ignorar)")
    ap.add_argument("--reenganche", type=float, default=3.0,
                    help="segundos quieto para continuar un evento anterior si el vehículo reaparece (hojas, tapados)")
    ap.add_argument("--sin-estabilizar", dest="sin_estabilizar", action="store_true",
                    help="desactivar la compensación del movimiento de cámara")
    ap.add_argument("--ignorar-izq", dest="ignorar_izq", type=float, default=0.0,
                    help="fracción izquierda de la imagen a ignorar al estimar el movimiento (ej. 0.2 si hay una baranda)")
    ap.add_argument("--fps", type=float, default=None,
                    help="forzar los cuadros por segundo del video (si el tiempo sale mal)")
    ap.add_argument("--solape", type=float, default=0.25, help="fracción mínima del vehículo dentro de la ciclovía")
    ap.add_argument("--saltar", type=int, default=3, help="procesar 1 de cada N cuadros (más rápido)")
    ap.add_argument("--modelo", default="yolov8n.pt", help="modelo YOLO (n = rápido, m = más preciso)")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--inicio", help='hora real del inicio del video, ej. "2026-10-10 08:30"')
    ap.add_argument("--salida", default="salida_video")
    ap.add_argument("--ver", action="store_true", help="mostrar el video con detecciones mientras procesa")
    ap.add_argument("--evaluar", metavar="CARPETA", help="calcula precisión/recall del eventos.csv validado")
    ap.add_argument("--perdidos", type=int, default=None, help="invasiones reales no detectadas (para recall)")
    args = ap.parse_args()

    if args.evaluar:
        evaluar(args.evaluar, args.perdidos)
        return
    if not args.video or not os.path.exists(args.video):
        sys.exit("Indica un video existente. Ej: python deteccion_video.py video.mp4 --calibrar --ancho 2 --largo 6")

    os.makedirs(args.salida, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    cuadros = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Video: {fps:.1f} cuadros/s, {cuadros:.0f} cuadros -> duración que usará el programa: "
          f"{cuadros / fps:.1f} s  (si no coincide con la real, usa --fps)")
    ok, primer = cap.read()
    if not ok:
        sys.exit("No se pudo leer el video.")
    alto, ancho_px = primer.shape[:2]
    datos = di.preparar_calibracion(args, primer, args.salida)
    ancho, largo = datos["ancho"], datos["largo"]
    H, Hinv = di.homografia(datos["puntos"], ancho, largo)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    inicio_dt = parsear_inicio(args.inicio)

    # La calibración queda en --salida (compartida entre videos grabados desde el mismo lugar);
    # los resultados de CADA video van en su propia subcarpeta para que no se pisen entre sí.
    carpeta = os.path.join(args.salida, os.path.splitext(os.path.basename(args.video))[0])
    os.makedirs(carpeta, exist_ok=True)
    for viejo in glob.glob(os.path.join(carpeta, "evento_*")):   # restos de una corrida anterior
        os.remove(viejo)
    print(f"Los resultados de este video se guardarán en: {carpeta}")

    print("Cargando modelo (la primera vez descarga los pesos)...")
    detectar = crear_tracker(args.modelo, args.conf, args.imgsz)
    print("Procesando video...")
    estab = None if args.sin_estabilizar else Estabilizador(primer, args.ignorar_izq)
    homs, stats = {}, {}
    eventos = procesar_video(args, H, Hinv, ancho, largo, detectar, cap, fps, carpeta, inicio_dt,
                             estab, homs, stats)
    cap.release()
    if estab is not None:
        print(f"Estabilización: la cámara se movió hasta {estab.max_desp:.0f} px respecto al primer cuadro; "
              f"cuadros sin poder ajustarse: {estab.fallos}")

    print("Generando clips e informe...")
    cortar_clips(args.video, eventos, Hinv, ancho, largo, carpeta, fps, (ancho_px, alto),
                 inicio_dt, args.saltar, homs)
    regenerar_fotos(args.video, eventos, ancho, largo, carpeta, inicio_dt)
    escribir_csv(eventos, carpeta, inicio_dt)
    escribir_informe(eventos, carpeta, inicio_dt, args)
    imprimir_diagnostico(stats, args, inicio_dt)
    print(f"\nListo. Eventos detectados: {len(eventos)}")
    print(f"Abre el informe: {os.path.join(carpeta, 'informe.html')}")


if __name__ == "__main__":
    main()
