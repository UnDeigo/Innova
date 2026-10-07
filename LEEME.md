# Prototipo con imagen: vehículo en ciclovía y distancia a ciclista

## Instalación (una vez)
    pip install -r requirements.txt

## 1) Calibrar (una vez por posición de cámara)
Mide con huincha un tramo de ciclovía: su ANCHO y su LARGO (ej. 2,0 m x 6,0 m).
    python deteccion_imagen.py foto.jpg --calibrar --ancho 2.0 --largo 6.0
Haz clic en las 4 esquinas del tramo EN ESTE ORDEN:
  1 cerca-izquierda, 2 cerca-derecha, 3 lejos-derecha, 4 lejos-izquierda. ENTER para guardar.
Las fotos siguientes deben tomarse con la cámara en el MISMO lugar y ángulo (trípode o apoyada).

## 2) Analizar
    python deteccion_imagen.py foto.jpg
    python deteccion_imagen.py carpeta_con_fotos/
Resultados en `salida_imagen/`: fotos anotadas, vehiculos.csv y pares.csv.

Veredicto de distancia:
  MENOS          incluso en el mejor caso, está a menos de 1,5 m
  OK             incluso en el peor caso, está a 1,5 m o más
  INDETERMINADO  el rango de error cruza los 1,5 m (la imagen sola no alcanza para decidir)

## 3) Medir qué tan bien funciona
1. En vehiculos.csv completa "en_ciclovia_real (si/no)".
2. En pares.csv completa "distancia_real_m" (medida con huincha al sacar la foto).
3. python deteccion_imagen.py --evaluar salida_imagen
Te da: % de aciertos en ciclovía, error medio en metros y si la distancia real cayó dentro del rango.

## Cómo hacer las fotos de prueba (seguridad primero)
Usen un estacionamiento o calle cerrada, nunca una calle con tráfico. Marquen con huincha
distancias reales de 0,5 / 1,0 / 1,5 / 2,0 / 3,0 m entre un auto detenido y una persona con bicicleta.
Tomen 2 o 3 fotos por distancia y anoten la medida.

---

# Video: vehículo detenido en la ciclovía (deteccion_video.py)

Usa la misma calibración en metros que el programa de imagen. La calibración se hace sobre el
primer cuadro del VIDEO (la cámara debe estar fija, en trípode o apoyada).

    python deteccion_video.py video.mp4 --calibrar --ancho 2.0 --largo 6.0 --ver

Después, para el mismo video o uno grabado desde el mismo lugar:

    python deteccion_video.py video.mp4 --inicio "2026-10-10 08:30"

Resultados de cada video en su propia subcarpeta: `salida_video/<nombre_del_video>/` con informe.html
(ábrelo en el navegador), eventos.csv, foto, recorte del vehículo y clip de cada evento. Así los videos
no se pisan entre sí. La calibración queda compartida en `salida_video/calibracion.json`. Si repites
el análisis del MISMO video, se reemplazan sus resultados anteriores.

Cómo decide: un vehículo con al menos 25% de su base dentro de la ciclovía que pasa más de
10 segundos SIN moverse más de 0,8 m se marca como invasión.

Parámetros útiles:
    --segundos 10    tiempo detenido para marcar invasión
    --mov-m 0.8      cuánto puede moverse (metros) y seguir contando como "detenido"
    --saltar 3       procesa 1 de cada 3 cuadros (más rápido; usa 1 para más precisión)
    --modelo yolov8m.pt   modelo más preciso pero más lento
    --ver            muestra el video con las detecciones mientras procesa (q para salir)
    --calibrar       rehacer la calibración (si cambiaste la cámara de lugar)

Medir qué tan bien funciona:
1. Abre eventos.csv y escribe si/no en "correcto (si/no)".
2. Cuenta a mano cuántas invasiones reales se le escaparon (N).
3. python deteccion_video.py --evaluar salida_video/nombre_del_video --perdidos N
