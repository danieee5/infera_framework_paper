# INFERA

INFERA es un conjunto reproducible de scripts para medir la energía consumida
por una GPU durante una conversación incremental con un modelo de lenguaje
autoalojado.

El experimento compara dos estrategias:

- **Historial completo:** cada petición conserva toda la conversación previa.
- **Compactación periódica:** cuando el prompt supera una regla de longitud,
  el modelo genera un resumen y continúa desde ese estado reducido.

Para cada estrategia, INFERA registra energía en joules, potencia, duración,
tokens de entrada y salida, eventos de compactación y éxito programático de
las tareas. No es necesario leer el paper para utilizar el repositorio.

Proyecto académico de Daniela Mora, Universidad de Especialidades Espíritu
Santo, Ecuador, 2026.

## Qué puedes hacer

### Medir tu propia sesión

Esta es la ruta principal. Puedes sustituir la base de conocimiento, las
tareas, los modelos, la cantidad de réplicas y la regla de compactación.
Necesitas Linux, una GPU NVIDIA dedicada, CUDA, NVML y dos representaciones
compatibles del modelo que quieras comparar.

El flujo es:

```text
configuración + base de conocimiento
              ↓
       ejecución en GPU
              ↓
     archivos JSONL crudos
              ↓
      tablas + figuras + informe
```

Empieza con la [guía desde cero](./infera/GUIA_DESDE_CERO.md).

### Auditar el estudio publicado

El repositorio incluye, como conjunto de referencia, las doce sesiones
utilizadas en el paper. Esta ruta no vuelve a ejecutar el modelo: verifica las
huellas de los JSONL y recalcula las cifras, tablas y figuras a partir de esas
mediciones.

Empieza con la [guía de auditoría](./docs/REPRODUCCION.md).

## Inicio rápido para una medición propia

Desde la raíz del repositorio:

```bash
cd infera
cp config/experiment.env.example config/experiment.env
cp config/session_tasks.example.json config/mi_sesion.json
```

Después:

1. Edita `config/mi_sesion.json` con tus tareas y reglas de validación.
2. Sustituye o adapta la base ficticia de `kb/`.
3. Edita `config/experiment.env` con tus modelos, etiqueta y parámetros.
4. Valida la configuración con el tokenizador real.
5. Ejecuta primero una prueba de humo.
6. Ejecuta la corrida completa y revisa `results/runs/`.

Los comandos completos y las comprobaciones están en
[`infera/GUIA_DESDE_CERO.md`](./infera/GUIA_DESDE_CERO.md).

## Qué genera una corrida

Cada combinación de representación, estrategia y réplica produce un JSONL.
Al finalizar, `analyze_results.py` genera:

- tablas normalizadas por sesión y tarea;
- totales de energía por condición;
- contabilidad del costo de los resúmenes;
- diferencia acumulada entre estrategias;
- tokens de entrada y salida;
- puntaje programático;
- tres figuras PNG;
- un informe legible y un manifiesto de procedencia.

Consulta [la guía de salidas](./docs/SALIDAS.md) para identificar cada archivo.

## Alcance configurable

Sin modificar el código puedes cambiar:

- rutas o identificadores de los modelos FP16 y AWQ;
- base de conocimiento;
- secuencia y cantidad de tareas;
- reglas programáticas de verificación;
- cantidad de réplicas;
- presupuesto de contexto;
- longitud máxima de salida;
- regla de activación de la compactación.

El código compara historial completo contra compactación periódica activada
por longitud. RAG, memoria externa, otro compresor o una política distinta
requieren adaptar y volver a validar el runner.

## Organización

- [`infera/`](./infera/): código, configuración, base ficticia, ejecución,
  análisis y resultados de referencia.
- [`docs/`](./docs/): instalación, configuración, reproducción, salidas,
  limitaciones y mapa para el anexo.
- [`requirements-gpu.txt`](./requirements-gpu.txt): entorno para ejecutar la
  medición con GPU.
- [`requirements.txt`](./requirements.txt): entorno mínimo para analizar el
  conjunto de referencia sin GPU.

## Resultado de referencia

El estudio incluido comparó Llama 3.1 8B Instruct bajo FP16 y AWQ en una RTX
4090. Se ejecutaron tres réplicas de historial completo y compactación para
cada representación, con 29 tareas por sesión.

En esa configuración, la política de tres compactaciones no recuperó la
energía empleada para producir los resúmenes dentro del horizonte observado.
La conclusión no implica que compactar siempre sea ineficiente ni identifica
un umbral universal.

## Documentación

- [Instalación](./docs/INSTALACION.md)
- [Guía desde cero](./infera/GUIA_DESDE_CERO.md)
- [Configuración](./docs/CONFIGURACION.md)
- [Auditar el conjunto de referencia](./docs/REPRODUCCION.md)
- [Archivos de salida](./docs/SALIDAS.md)
- [Limitaciones](./docs/LIMITACIONES.md)
- [Mapa para el anexo](./docs/ANEXO_REPOSITORIO.md)

El repositorio todavía no declara una licencia de reutilización ni una forma
de citación definitiva. Esos archivos deben añadirse después de confirmar las
condiciones institucionales y los metadatos finales.
