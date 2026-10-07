#!/usr/bin/env python3
"""
Prototipo con IMAGEN: ¿hay un vehículo dentro de la ciclovía? ¿está a menos de 1,5 m de un ciclista?

Cómo funciona:
  1. Calibras UNA vez por posición de cámara: haces clic en las 4 esquinas de un tramo de
     ciclovía (rectángulo en el suelo) e indicas su ancho y largo reales en metros.
     Con eso el programa convierte píxeles del suelo en metros (homografía).
  2. YOLO detecta vehículos y bicicletas en la foto.
  3. Se ubica el borde inferior de cada vehículo (donde apoya en el suelo) y el punto de
     apoyo de cada bicicleta, y se calcula:
        - qué porcentaje del vehículo está dentro de la franja de la ciclovía
        - la distancia en metros entre vehículo y ciclista, con un rango de incertidumbre
  4. Veredicto de distancia: MENOS (< límite) / OK (>= límite) / INDETERMINADO (el rango cruza el límite).

Ejemplos:
  python deteccion_imagen.py foto.jpg --calibrar --ancho 2.0 --largo 6.0
  python deteccion_imagen.py foto.jpg                      # reutiliza la calibración
  python deteccion_imagen.py carpeta_con_fotos/            # varias fotos (misma cámara)
  python deteccion_imagen.py --evaluar salida              # compara con tus mediciones reales
"""
import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

CLASES_VEHICULO = {2: "auto", 3: "moto", 5: "bus", 7: "camion"}
CLASE_BICI = 1          # 'bicycle' en COCO
MUESTRAS = 11           # puntos a lo largo del borde inferior de cada vehículo
EXTENSION_M = 3.0       # la franja de ciclovía se extiende 3 m más allá del tramo calibrado


# ----------------------------------------------------------------------------
# Calibración (píxeles -> metros)
# ----------------------------------------------------------------------------
def elegir_puntos(img):
    """4 clics: cerca-izq, cerca-der, lejos-der, lejos-izq (esquinas de un tramo de ciclovía)."""
    pts = []
    base = img.copy()
    textos = ["1: esquina CERCANA izquierda", "2: esquina CERCANA derecha",
              "3: esquina LEJANA derecha", "4: esquina LEJANA izquierda", "ENTER para guardar"]

    def redibujar():
        v = base.copy()
        for i, p in enumerate(pts):
            cv2.circle(v, p, 6, (0, 0, 255), -1)
            cv2.putText(v, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(pts) > 1:
            cv2.polylines(v, [np.array(pts)], len(pts) == 4, (0, 255, 255), 2)
        cv2.putText(v, textos[min(len(pts), 4)] + "   (Z: deshacer, ESC: cancelar)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Calibracion", v)

    def click(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            redibujar()

    cv2.namedWindow("Calibracion")
    cv2.setMouseCallback("Calibracion", click)
    redibujar()
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (13, 10) and len(pts) == 4:
            break
        if k == ord("z") and pts:
            pts.pop()
            redibujar()
        if k == 27:
            sys.exit("Cancelado.")
    cv2.destroyWindow("Calibracion")
    cv2.waitKey(1)
    return pts


def preparar_calibracion(args, img, carpeta):
    ruta = os.path.join(carpeta, "calibracion.json")
    if args.puntos or args.calibrar or not os.path.exists(ruta):
        if args.ancho is None or args.largo is None:
            sys.exit("Para calibrar indica --ancho y --largo (metros reales del tramo). "
                     "Ej: --calibrar --ancho 2.0 --largo 6.0")
        if args.puntos:
            pts = [tuple(int(v) for v in p.split(",")) for p in args.puntos.split(";")]
            if len(pts) != 4:
                sys.exit("--puntos necesita exactamente 4 puntos")
        else:
            pts = elegir_puntos(img)
        datos = {"puntos": pts, "ancho": args.ancho, "largo": args.largo}
        with open(ruta, "w") as f:
            json.dump(datos, f)
    else:
        with open(ruta) as f:
            datos = json.load(f)
    return datos


def homografia(pts, ancho, largo):
    src = np.float32(pts)
    dst = np.float32([[0, 0], [ancho, 0], [ancho, largo], [0, largo]])
    return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)


def transformar(H, puntos):
    p = np.float32(puntos).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


# ----------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------
def crear_detector(modelo_nombre, conf, imgsz):
    from ultralytics import YOLO
    modelo = YOLO(modelo_nombre)
    clases = list(CLASES_VEHICULO) + [CLASE_BICI]

    def detectar(img):
        r = modelo(img, conf=conf, imgsz=imgsz, classes=clases, verbose=False)[0]
        salida = []
        if r.boxes is None:
            return salida
        for caja, c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.int().cpu().tolist(),
                              r.boxes.conf.cpu().tolist()):
            x1, y1, x2, y2 = (float(v) for v in caja)
            salida.append((c, p, x1, y1, x2, y2))
        return salida

    return detectar


# ----------------------------------------------------------------------------
# Análisis geométrico (no depende del modelo: se puede probar con cajas inventadas)
# ----------------------------------------------------------------------------
def borde_inferior(x1, x2, y):
    return np.stack([np.linspace(x1, x2, MUESTRAS), np.full(MUESTRAS, y)], axis=1)


def analizar(detecciones, H, ancho, largo, limite=1.5, solape_min=0.25,
             error_px=4.0, max_par_m=6.0):
    vehiculos, ciclistas = [], []
    for c, p, x1, y1, x2, y2 in detecciones:
        if c in CLASES_VEHICULO:
            vehiculos.append({"id": f"V{len(vehiculos) + 1}", "clase": CLASES_VEHICULO[c],
                              "caja": (x1, y1, x2, y2)})
        elif c == CLASE_BICI:
            ciclistas.append({"id": f"C{len(ciclistas) + 1}", "caja": (x1, y1, x2, y2)})

    for v in vehiculos:
        x1, y1, x2, y2 = v["caja"]
        mundo = transformar(H, borde_inferior(x1, x2, y2))
        dentro = ((mundo[:, 0] >= 0) & (mundo[:, 0] <= ancho) &
                  (mundo[:, 1] >= -EXTENSION_M) & (mundo[:, 1] <= largo + EXTENSION_M))
        v["pct_ciclovia"] = float(dentro.mean())
        v["en_ciclovia"] = v["pct_ciclovia"] >= solape_min

    pares = []
    for v in vehiculos:
        vx1, vy1, vx2, vy2 = v["caja"]
        dv = max(error_px, 0.03 * (vy2 - vy1))
        for c in ciclistas:
            cx1, cy1, cx2, cy2 = c["caja"]
            dc = max(error_px, 0.03 * (cy2 - cy1))
            distancias = []
            nominal, mejor_idx = None, 0
            for ov in (-dv, 0.0, dv):
                base_v = transformar(H, borde_inferior(vx1, vx2, vy2 + ov))
                for oc in (-dc, 0.0, dc):
                    pc = transformar(H, [((cx1 + cx2) / 2.0, cy2 + oc)])[0]
                    d = np.linalg.norm(base_v - pc, axis=1)
                    distancias.append(float(d.min()))
                    if ov == 0.0 and oc == 0.0:
                        nominal, mejor_idx = float(d.min()), int(d.argmin())
            if nominal > max_par_m:
                continue
            dmin, dmax = min(distancias), max(distancias)
            veredicto = "MENOS" if dmax < limite else ("OK" if dmin >= limite else "INDETERMINADO")
            xs = np.linspace(vx1, vx2, MUESTRAS)
            pares.append({"vehiculo": v["id"], "ciclista": c["id"], "d_min": dmin,
                          "d_med": nominal, "d_max": dmax, "veredicto": veredicto,
                          "punto_vehiculo": (float(xs[mejor_idx]), float(vy2)),
                          "punto_ciclista": ((cx1 + cx2) / 2.0, float(cy2))})
    return vehiculos, ciclistas, pares


# ----------------------------------------------------------------------------
# Dibujo del resultado
# ----------------------------------------------------------------------------
def texto(img, s, org, color, escala):
    (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, escala, 2)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + 4), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 2)


def extension_dibujable(Hinv, ancho, largo):
    """Cuánto extender la franja al dibujarla sin que la perspectiva se 'dé vuelta'."""
    w_ref = Hinv[2, 0] * ancho / 2 + Hinv[2, 1] * largo / 2 + Hinv[2, 2]
    signo = 1.0 if w_ref >= 0 else -1.0

    def valido(y):
        return all(signo * (Hinv[2, 0] * x + Hinv[2, 1] * y + Hinv[2, 2]) > 0.1 * abs(w_ref)
                   for x in (0.0, ancho))

    def buscar(direccion):
        e = EXTENSION_M
        while e > 0:
            y = -e if direccion == "cerca" else largo + e
            if valido(y):
                return e
            e -= 0.25
        return 0.0

    return buscar("cerca"), buscar("lejos")


def dibujar(img, vehiculos, ciclistas, pares, Hinv, ancho, largo, limite):
    out = img.copy()
    escala = max(0.5, img.shape[1] / 1600.0)
    e_cerca, e_lejos = extension_dibujable(Hinv, ancho, largo)
    franja = transformar(Hinv, [[0, -e_cerca], [ancho, -e_cerca],
                                [ancho, largo + e_lejos], [0, largo + e_lejos]])
    capa = out.copy()
    cv2.fillPoly(capa, [franja.astype(np.int32)], (255, 200, 0))
    out = cv2.addWeighted(capa, 0.25, out, 0.75, 0)
    cv2.polylines(out, [franja.astype(np.int32)], True, (255, 200, 0), 2)

    ver_por_vehiculo = {}
    for p in pares:
        orden = {"OK": 0, "INDETERMINADO": 1, "MENOS": 2}
        actual = ver_por_vehiculo.get(p["vehiculo"], "OK")
        if orden[p["veredicto"]] > orden[actual]:
            ver_por_vehiculo[p["vehiculo"]] = p["veredicto"]

    for v in vehiculos:
        x1, y1, x2, y2 = (int(t) for t in v["caja"])
        ver = ver_por_vehiculo.get(v["id"], "OK")
        if v["en_ciclovia"]:
            color, etiqueta = (0, 0, 255), f'{v["clase"]} EN CICLOVIA {v["pct_ciclovia"]:.0%}'
        elif ver == "MENOS":
            color, etiqueta = (0, 128, 255), f'{v["clase"]} < {limite:g} m'
        elif ver == "INDETERMINADO":
            color, etiqueta = (0, 220, 255), f'{v["clase"]} distancia dudosa'
        else:
            color, etiqueta = (0, 200, 0), v["clase"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        texto(out, f'{v["id"]} {etiqueta}', (x1, max(y1 - 10, 25)), color, escala)

    for c in ciclistas:
        x1, y1, x2, y2 = (int(t) for t in c["caja"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 120, 0), 3)
        texto(out, c["id"] + " ciclista", (x1, max(y1 - 10, 25)), (255, 120, 0), escala)

    colores = {"MENOS": (0, 128, 255), "INDETERMINADO": (0, 220, 255), "OK": (0, 200, 0)}
    for p in pares:
        a = tuple(int(t) for t in p["punto_vehiculo"])
        b = tuple(int(t) for t in p["punto_ciclista"])
        cv2.line(out, a, b, colores[p["veredicto"]], 3)
        medio = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        texto(out, f'{p["d_min"]:.1f}-{p["d_max"]:.1f} m', medio, colores[p["veredicto"]], escala)
    return out


# ----------------------------------------------------------------------------
# Evaluación contra mediciones reales
# ----------------------------------------------------------------------------
def a_numero(s):
    try:
        return float(str(s).replace(",", ".").strip())
    except ValueError:
        return None


def si_no(s):
    s = str(s or "").strip().lower()
    if s in ("si", "sí", "s", "1", "true"):
        return True
    if s in ("no", "n", "0", "false"):
        return False
    return None


def evaluar(carpeta, limite):
    rv = os.path.join(carpeta, "vehiculos.csv")
    rp = os.path.join(carpeta, "pares.csv")
    print("=== ¿Detecta bien cuando un vehículo está en la ciclovía? ===")
    tp = fp = fn = tn = 0
    with open(rv, encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            real = si_no(fila.get("en_ciclovia_real (si/no)"))
            if real is None:
                continue
            pred = si_no(fila["en_ciclovia"])
            tp += real and pred
            fp += (not real) and pred
            fn += real and (not pred)
            tn += (not real) and (not pred)
    n = tp + fp + fn + tn
    if n:
        print(f"Casos evaluados: {n}  | aciertos: {(tp + tn) / n:.0%}  "
              f"| falsos positivos: {fp}  | falsos negativos: {fn}")
    else:
        print("Completa la columna 'en_ciclovia_real (si/no)' en vehiculos.csv")

    print("\n=== ¿Qué tan buena es la distancia estimada? ===")
    errores, dentro, aciertos, indet, total = [], 0, 0, 0, 0
    with open(rp, encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            real = a_numero(fila.get("distancia_real_m"))
            if real is None:
                continue
            total += 1
            dmed, dmin, dmax = (float(fila[k]) for k in ("d_med_m", "d_min_m", "d_max_m"))
            errores.append(abs(dmed - real))
            dentro += dmin <= real <= dmax
            if fila["veredicto"] == "INDETERMINADO":
                indet += 1
            elif (fila["veredicto"] == "MENOS") == (real < limite):
                aciertos += 1
    if total:
        print(f"Casos con distancia real: {total}")
        print(f"Error medio absoluto: {np.mean(errores):.2f} m  (máximo: {max(errores):.2f} m)")
        print(f"La distancia real cayó dentro del rango estimado: {dentro / total:.0%}")
        decididos = total - indet
        print(f"Veredicto MENOS/OK correcto: {aciertos}/{decididos}  "
              f"(INDETERMINADO en {indet} casos)")
    else:
        print("Completa la columna 'distancia_real_m' en pares.csv (medida con huincha).")


# ----------------------------------------------------------------------------
def listar_imagenes(ruta):
    if os.path.isdir(ruta):
        archivos = []
        for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
            archivos += glob.glob(os.path.join(ruta, f"*.{ext}"))
        return sorted(archivos)
    return [ruta]


def procesar_imagenes(args, detectar):
    imagenes = listar_imagenes(args.imagen)
    if not imagenes:
        sys.exit("No encontré imágenes.")
    os.makedirs(args.salida, exist_ok=True)
    primera = cv2.imread(imagenes[0])
    datos = preparar_calibracion(args, primera, args.salida)
    H, Hinv = homografia(datos["puntos"], datos["ancho"], datos["largo"])

    filas_v, filas_p = [], []
    for ruta in imagenes:
        img = cv2.imread(ruta)
        if img is None:
            print(f"No pude leer {ruta}")
            continue
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        dets = detectar(img)
        vehs, cics, pares = analizar(dets, H, datos["ancho"], datos["largo"], args.limite,
                                     args.solape, args.error_px, args.max_par)
        cv2.imwrite(os.path.join(args.salida, f"{nombre}_resultado.jpg"),
                    dibujar(img, vehs, cics, pares, Hinv, datos["ancho"], datos["largo"], args.limite))
        print(f"{nombre}: {len(vehs)} vehículo(s), {len(cics)} ciclista(s)")
        for v in vehs:
            print(f'  {v["id"]} {v["clase"]}: {v["pct_ciclovia"]:.0%} dentro de la ciclovía -> '
                  f'{"EN CICLOVIA" if v["en_ciclovia"] else "fuera"}')
            filas_v.append([nombre, v["id"], v["clase"], f'{v["pct_ciclovia"]:.2f}',
                            "si" if v["en_ciclovia"] else "no", ""])
        for p in pares:
            print(f'  {p["vehiculo"]} - {p["ciclista"]}: {p["d_min"]:.2f} a {p["d_max"]:.2f} m '
                  f'(estimado {p["d_med"]:.2f}) -> {p["veredicto"]}')
            filas_p.append([nombre, p["vehiculo"], p["ciclista"], f'{p["d_min"]:.2f}',
                            f'{p["d_med"]:.2f}', f'{p["d_max"]:.2f}', p["veredicto"], ""])

    with open(os.path.join(args.salida, "vehiculos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["imagen", "id", "clase", "fraccion_en_ciclovia", "en_ciclovia",
                    "en_ciclovia_real (si/no)"])
        w.writerows(filas_v)
    with open(os.path.join(args.salida, "pares.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["imagen", "vehiculo", "ciclista", "d_min_m", "d_med_m", "d_max_m",
                    "veredicto", "distancia_real_m"])
        w.writerows(filas_p)
    print(f"\nListo. Resultados en la carpeta '{args.salida}'.")


def main():
    ap = argparse.ArgumentParser(description="Detección de vehículos en ciclovía y distancia a ciclistas (imagen)")
    ap.add_argument("imagen", nargs="?", help="foto o carpeta con fotos de la MISMA posición de cámara")
    ap.add_argument("--calibrar", action="store_true", help="(re)hacer la calibración")
    ap.add_argument("--ancho", type=float, help="ancho real del tramo de ciclovía en metros")
    ap.add_argument("--largo", type=float, help="largo real del tramo calibrado en metros")
    ap.add_argument("--puntos", help='4 puntos manuales "x,y;x,y;x,y;x,y" (cerca-izq, cerca-der, lejos-der, lejos-izq)')
    ap.add_argument("--limite", type=float, default=1.5, help="distancia mínima legal en metros")
    ap.add_argument("--solape", type=float, default=0.25, help="fracción mínima del vehículo dentro de la ciclovía")
    ap.add_argument("--error-px", dest="error_px", type=float, default=4.0, help="error esperado de la caja en píxeles")
    ap.add_argument("--max-par", dest="max_par", type=float, default=6.0, help="ignorar pares a más de X metros")
    ap.add_argument("--modelo", default="yolov8m.pt", help="modelo YOLO (m = más preciso; n = más rápido)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--salida", default="salida_imagen")
    ap.add_argument("--evaluar", metavar="CARPETA", help="calcula el error usando tus mediciones reales")
    args = ap.parse_args()

    if args.evaluar:
        evaluar(args.evaluar, args.limite)
        return
    if not args.imagen or not os.path.exists(args.imagen):
        sys.exit("Indica una imagen o carpeta. Ej: python deteccion_imagen.py foto.jpg --calibrar --ancho 2 --largo 6")
    print("Cargando modelo (la primera vez descarga los pesos)...")
    procesar_imagenes(args, crear_detector(args.modelo, args.conf, args.imgsz))


if __name__ == "__main__":
    main()
