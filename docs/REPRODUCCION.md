# Auditar los resultados publicados

Esta ruta está destinada a revisores que quieren comprobar los cálculos del
paper sin alquilar una GPU.

Los JSONL ya contienen las mediciones físicas. El análisis no los reconstruye
ni simula: vuelve a calcular tablas, totales y figuras desde esos datos.

## 1. Instalar el analizador

Desde la raíz:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd infera
```

## 2. Verificar las mediciones

```bash
cd results/reference/raw
shasum -a 256 -c SHA256SUMS
cd ../../..
```

Los doce archivos deben indicar `OK`.

## 3. Regenerar

```bash
python analyze_results.py
```

La consola debe informar:

```json
{
  "ok": true,
  "rows": 366,
  "runs": 12,
  "out": "results/reference/expected"
}
```

El programa genera integridad, filas normalizadas, resúmenes por corrida y
condición, contabilidad de resúmenes, ciclos, diferencia acumulada, tokens,
puntaje programático, tres figuras, un informe y un manifiesto.

## Resultados principales esperados

- AWQ con historial completo: 11.583,78 J.
- AWQ con compactación: 15.051,62 J.
- Diferencia AWQ: +3.467,84 J.
- FP16 con historial completo: 15.671,55 J.
- FP16 con compactación: 22.762,85 J.
- Diferencia FP16: +7.091,30 J.

## Auditoría aislada

Para no reemplazar la copia esperada:

```bash
python analyze_results.py \
  --source results/reference/raw \
  --out /tmp/infera_reference_audit
```

Después puedes comparar esa carpeta con `results/reference/expected/`.

## Repetir el experimento físico

Repetir el modelo en GPU es otra tarea. Consulta
[`../infera/GUIA_DESDE_CERO.md`](../infera/GUIA_DESDE_CERO.md).

No se espera igualdad exacta de joules si cambian GPU, controladores,
temperatura, versión de vLLM, checkpoints o procesos concurrentes.
