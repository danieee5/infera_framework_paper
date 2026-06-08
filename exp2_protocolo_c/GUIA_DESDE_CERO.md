# GUÍA DESDE CERO — Dejar corriendo el Protocolo C toda la noche

Esta guía asume que ya tienes el pod de RunPod (RTX 4090) y los modelos en `/models/`.
Síguela en orden. No necesitas decidir nada: las decisiones ya están tomadas.

---

## DECISIÓN 1: ¿repo nuevo o el mismo?

**El MISMO repo de EXP1** (`titan_framework_paper`), en una **rama nueva** y una **carpeta nueva**.
Razón: EXP1 es tu piloto (lo citamos así en el paper); mantener todo junto da continuidad y no pierdes tus 243 corridas.

---

## DECISIÓN 2: estructura final del repo

```
titan_framework_paper/
├── (todo lo de EXP1, intacto)              <- branch main / pilot-run
└── exp2_protocolo_c/                       <- carpeta nueva (en branch nueva)
    ├── kb/
    │   ├── vigia_kb.md
    │   ├── permisos_medicos.csv
    │   └── inventario_uniformes.csv
    ├── session_tasks.json
    ├── infera_kb.py
    ├── infera_quality.py
    ├── infera_compaction.py
    ├── infera_session_runner.py
    ├── infera_analysis.py
    ├── gpu_power_monitor.py        <- COPIА el de EXP1 aquí (lo importa el runner)
    ├── setup_infera.sh
    ├── run_all.sh                  <- corrida manual de UNA cuantización
    ├── overnight.sh                <- corrida DESATENDIDA de toda la noche
    ├── README.md
    ├── GUIA_DESDE_CERO.md          <- este archivo
    ├── paper/
    │   └── INFERA_paper_C.md
    └── results/                    <- se crea solo al correr
```

---

## PASO 0 — En tu computador: armar la carpeta y subir a GitHub

```bash
# 1. Clona tu repo (si no lo tienes local) o entra a él
git clone https://github.com/danieee5/titan_framework_paper.git
cd titan_framework_paper

# 2. Crea la rama nueva
git checkout -b exp2-protocolo-c

# 3. Copia TODO lo que te entregué dentro de exp2_protocolo_c/
#    (descomprime el paquete infera_c y renómbralo a exp2_protocolo_c, o copia archivo por archivo)

# 4. COPIA tu gpu_power_monitor.py de EXP1 dentro de exp2_protocolo_c/
cp gpu_power_monitor.py exp2_protocolo_c/    # ajusta la ruta de origen si está en otra carpeta

# 5. Sube
git add exp2_protocolo_c/
git commit -m "EXP2 Protocolo C: framework sobre energia-calidad + compactacion"
git push -u origin exp2-protocolo-c
```

---

## PASO 1 — En el pod de RunPod: traer el código

```bash
cd /workspace
# si ya tienes el repo clonado en el pod:
cd titan_framework_paper && git fetch && git checkout exp2-protocolo-c && git pull
# si no:
git clone https://github.com/danieee5/titan_framework_paper.git
cd titan_framework_paper && git checkout exp2-protocolo-c

cd exp2_protocolo_c
```

> Verifica que `gpu_power_monitor.py` esté en esta carpeta. Si no: `cp /ruta/a/gpu_power_monitor.py .`

---

## PASO 2 — Instalar el entorno (UNA sola vez)

```bash
bash setup_infera.sh
source /workspace/venv/bin/activate
```
Al final imprime la versión de torch, vLLM y si NVML está OK. Si NVML falla, las medidas de energía saldrán en cero: revisa que el pod tenga acceso a la GPU.

---

## PASO 3 — Prueba de humo (2 minutos, opcional pero recomendado)

```bash
# arma el contexto fijo
python infera_kb.py kb | tail -3

# levanta vLLM FP16 en segundo plano para una prueba corta
nohup python -m vllm.entrypoints.openai.api_server \
  --model /models/llama3.1-8b-instruct --dtype float16 \
  --max-model-len 8192 --port 8000 > vllm_test.log 2>&1 &

# espera ~1-2 min y comprueba que responde
sleep 90 && curl -s http://localhost:8000/v1/models | head -c 200

# corre UNA sesión naive de prueba
python infera_session_runner.py \
  --vllm-url http://localhost:8000/v1/chat/completions \
  --model /models/llama3.1-8b-instruct \
  --quant FP16 --arm naive --rep 1 \
  --kb-dir kb --tasks session_tasks.json \
  --out results/PRUEBA_fp16_naive.jsonl

# si ves 16 líneas con ctx=, E= y Q=, FUNCIONA. baja vLLM:
pkill -f vllm.entrypoints.openai.api_server
```

---

## PASO 4 — LA CORRIDA NOCTURNA (lo que querías)

**Clave: usa `tmux`.** Si corres directo por SSH y se te cae la conexión, se mata el experimento. `tmux` lo mantiene vivo aunque cierres la laptop.

```bash
# 1. abre una sesión tmux
tmux new -s infera

# 2. (dentro de tmux) activa el entorno
source /workspace/venv/bin/activate
cd /workspace/titan_framework_paper/exp2_protocolo_c

# 3. lanza la corrida desatendida (FP16 + AWQ, naive + compaction, 3 reps c/u)
bash overnight.sh

# 4. DESCONECTA sin matar nada:  pulsa  Ctrl+b  y luego  d   (detach)
#    ya puedes cerrar la laptop. El experimento sigue.
```

Para volver a mirar (desde cualquier máquina, re-SSH al pod):
```bash
tmux attach -t infera        # vuelves a ver el progreso en vivo
# detach otra vez con Ctrl+b  d
```

Qué hace `overnight.sh` solo: sirve FP16 → corre naive×3 y compaction×3 → baja vLLM → sirve AWQ → repite → corre el análisis y genera las figuras. Todo queda en `results/` y en un log `overnight_FECHA.log`.

Duración estimada: 2 cuantizaciones × 2 brazos × 3 reps × (16 tareas + cooling) ≈ varias horas; entra holgado en una noche.

---

## PASO 5 — En la mañana

```bash
tmux attach -t infera                       # ¿terminó?  busca "FIN" al final
ls results/                                 # deben estar los run_*.jsonl
ls results/analysis/                        # envelope_FP16.png, envelope_AWQ.png, recovery_*.csv
cat results/analysis/recovery_naive_vs_compaction.csv
```

Mira el log para el **codo** detectado:
```bash
grep CODO overnight_*.log
```
- Si aparece un codo en el brazo naive FP16 (ej. `~5200 tok`), ese es tu hallazgo. Si el umbral provisional (5000) quedó lejos del codo real, **re-corre solo el brazo compaction** con el umbral calibrado:
  ```bash
  # (con vLLM FP16 servido)
  ./run_all.sh FP16 /models/llama3.1-8b-instruct 3 <umbral_del_codo>
  ```
- Si el codo **no aparece** ("no detectado"): la sesión es muy corta para que el rot muerda en 8k. Duplica el bloque de tareas RECALL/DRAFT en `session_tasks.json` (copia T11–T16 con nuevos ids T17–T22) y vuelve a correr. Esto fuerza más acumulación.

Sube los resultados:
```bash
git add results/ && git commit -m "EXP2 Protocolo C: resultados primera tanda" && git push
```

---

## TROUBLESHOOTING

- **vLLM no levanta / se omite una cuantización:** mira `vllm_FP16.log` o `vllm_AWQ.log`. Causa común: VRAM no liberada del modelo anterior → `overnight.sh` ya espera 15 s; si persiste, sube ese valor en `stop_vllm`.
- **Energía en 0.0 / NVML no disponible:** el pod no expone NVML. Reinicia el pod o verifica `nvidia-smi`.
- **OOM al servir:** baja `--max-model-len` (ej. 4096) o usa AWQ, que ocupa menos VRAM.
- **Calidad siempre 0 en tareas de rota:** el modelo no devolvió JSON parseable; es un resultado válido (calidad baja), pero si quieres revisar, mira el texto en el `.jsonl`.
- **Se cayó el SSH y no usaste tmux:** el job murió. Por eso el Paso 4 insiste en tmux.

---

## RESUMEN DE COMANDOS (chuleta)

```bash
# una vez
bash setup_infera.sh && source /workspace/venv/bin/activate

# noche
tmux new -s infera
source /workspace/venv/bin/activate && cd /workspace/titan_framework_paper/exp2_protocolo_c
bash overnight.sh
# Ctrl+b  d   (detach, cierra laptop)

# mañana
tmux attach -t infera
cat results/analysis/recovery_naive_vs_compaction.csv
grep CODO overnight_*.log
```
