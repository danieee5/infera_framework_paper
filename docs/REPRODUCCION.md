# Auditar el experimento publicado

Esta ruta está destinada a quien desea comprobar datos y cálculos del paper
sin alquilar una GPU. No reconstruye ni simula las mediciones físicas: parte
de las trazas NVML persistidas en los JSONL.

## 1. Requisitos

La auditoría y la reanálisis utilizan Python 3.10 o superior y la biblioteca
estándar. Desde la raíz del repositorio:

```bash
python3 --version
```

Matplotlib solo es necesario para regenerar las figuras.

## 2. Identificar las 18 sesiones

Los crudos están en:

```text
experimentos/experimento_principal/evidencia/raw/
```

Debe haber seis condiciones y tres repeticiones instrumentales:

```text
AWQ  × completo, resumen, descarte × rep1, rep2, rep3
FP16 × completo, resumen, descarte × rep1, rep2, rep3
```

Cada JSONL tiene un manifiesto de sesión asociado. No mezcles archivos de otra
corrida en esta carpeta.

## 3. Ejecutar la auditoría completa

Utiliza una ruta de reanálisis que todavía no exista:

```bash
python3 infera/audita_paquete_tres_brazos.py \
  --campaign experimentos/experimento_principal/evidencia \
  --archive experimentos/experimento_principal/paquete_preservado/experimento_principal_evidencia.tar.gz.bin \
  --checksum experimentos/experimento_principal/paquete_preservado/experimento_principal_evidencia.tar.gz.sha256 \
  --reanalysis /tmp/infera_reanalysis
```

La salida aceptada debe incluir:

```json
{
  "ok": true,
  "status": "complete",
  "exit_code": 0,
  "artifacts_checked": {
    "raws": 18,
    "session_manifests": 18,
    "analysis": 6,
    "logs": 11
  },
  "tasks": 29,
  "sessions": 18,
  "partials": 0,
  "reanalysis_matches_download": true
}
```

## 4. Qué comprueba

El auditor:

- verifica el SHA-256 y tamaño de cada artefacto declarado;
- exige finalización exitosa, 18 sesiones y cero archivos parciales;
- comprueba el escenario de 29 tareas y la base sintética;
- valida los manifiestos y las trazas NVML;
- confirma que el paquete preservado coincide archivo por archivo con
  `evidencia/`;
- reubica copias temporales de preflight, manifiesto y raws;
- reintegra cada traza por la regla trapezoidal;
- vuelve a producir los cinco resultados sustantivos y exige igualdad byte a
  byte con la descarga.

La carpeta pública se llama `evidencia`, mientras el tar conserva su
identificador original con marca temporal. El nombre externo no interviene en
la identidad de los 56 archivos relativos.

## 5. Qué no comprueba

La auditoría no demuestra:

- que los joules sean universales para otras GPU o versiones;
- que tres pasadas sean réplicas independientes de calidad;
- que el disparador de 4.500 tokens o `K=4` sean óptimos;
- que el puntaje programático equivalga a evaluación humana;
- que una política sea superior fuera de las 29 tareas observadas.

## 6. Regenerar figuras

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python3 infera/figuras_tres_brazos.py \
  --campaign experimentos/experimento_principal/evidencia \
  --output /tmp/infera_figuras
```

El comando crea cuatro PNG, cuatro PDF y
`manifiesto_figuras.json`. Este manifiesto registra hashes de fuentes y
derivados. La salida falla si el directorio ya existe para evitar
sobrescrituras silenciosas.

## 7. Repetir la medición física

Volver a medir requiere GPU y constituye una corrida nueva. Consulta
[Instalación](./INSTALACION.md), [Configuración](./CONFIGURACION.md) y la
[guía del runner](../infera/README.md).

No se espera igualdad exacta de joules si cambian GPU, controlador,
temperatura, checkpoints, vLLM o procesos concurrentes.
