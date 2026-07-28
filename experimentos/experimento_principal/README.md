# Experimento principal

## Diseño

La unidad energética es la sesión completa de 29 tareas. Se registraron las
seis condiciones del diseño y tres repeticiones instrumentales de cada una:

| Papel | Variable | Valores | Función en el diseño |
|---|---|---|---|
| Independiente | Política de historial | completo, resumen, descarte | Define qué historial llega a cada petición |
| Independiente | Representación | AWQ, FP16 | Define la representación numérica servida |
| Dependiente | Energía de sesión | joules | Suma end-to-end de tareas e intervenciones |
| Dependiente | Cumplimiento programático | 0 a 1 | Verifica reglas predeclaradas, no calidad humana integral |
| Controlada | Secuencia | 29 tareas fijas | Mantiene el mismo recorrido lógico |
| Controlada | Repetición | 1, 2, 3 | Comprueba estabilidad instrumental local |

`completo` conserva todos los pares usuario/asistente. `resumen` genera un
relevo cuando el prompt supera 4.500 tokens. `descarte` aplica un recorte local
y conserva los últimos `K=4` pares completos. Tanto el disparador como `K`
fueron decisiones operativas, no óptimos estimados.

## Estructura

```text
experimento_principal/
├── evidencia/             # 56 archivos originales, sin reserializar
│   ├── raw/               # 18 JSONL y 18 manifiestos de sesión
│   ├── analisis/          # tablas, informe y manifiesto derivados
│   ├── logs/              # preflight, launcher, sesiones y vLLM
│   ├── preflight.json
│   └── manifiesto_campana.json
├── paquete_preservado/    # contenedor exacto y SHA-256 externo
└── figuras/               # derivados regenerables y su manifiesto
```

Los nombres de `raw/` identifican sin una tabla auxiliar la representación,
la política y la repetición:

```text
run_AWQ_completo_rep1.jsonl
run_AWQ_resumen_rep1.jsonl
run_AWQ_descarte_rep1.jsonl
run_FP16_completo_rep1.jsonl
...
```

Cada manifiesto de sesión tiene el mismo nombre que su JSONL seguido de
`.manifiesto.json`.

## Qué es evidencia y qué es cálculo

- Evidencia registrada: JSONL, manifiestos de sesión, preflight y logs.
- Cálculo reproducible: integración de trazas NVML, agregados por sesión,
  efectos por tarea y figuras.
- Interpretación: el texto del paper; no está codificado en los raws.

`evidencia/` es la carpeta descargada de RunPod con un nombre externo estable.
Los metadatos internos conservan rutas absolutas y nombres de scripts de la
ejecución original porque forman parte de la procedencia. El auditor reubica
copias temporales para recalcular, nunca edita esta evidencia.

## Comprobar sin GPU

Desde la raíz del repositorio:

```bash
python3 infera/audita_paquete_tres_brazos.py \
  --campaign experimentos/experimento_principal/evidencia \
  --archive experimentos/experimento_principal/paquete_preservado/experimento_principal_evidencia.tar.gz.bin \
  --checksum experimentos/experimento_principal/paquete_preservado/experimento_principal_evidencia.tar.gz.sha256 \
  --reanalysis /tmp/infera_reanalysis
```

El paquete binario y la carpeta deben contener exactamente los mismos 56
archivos relativos. El SHA-256 esperado del paquete es:

```text
cd37010cae57f97012cb2bc8fc0dd40e6743736354bc6d8ff57e42352ae57c1d
```

Consulta la [guía detallada](../../docs/REPRODUCCION.md) para interpretar cada
control.
