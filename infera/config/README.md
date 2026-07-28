# Configuración

- `reference/session_tasks.json`: secuencia exacta de 29 tareas publicada.
- `reference/experiment.env.example`: parámetros del experimento principal;
  crea `config/experiment.env` y edita solo las rutas locales de los modelos.
- `session_tasks.example.json`: ejemplo mínimo para diseñar otro caso.
- `experiment.env.example`: plantilla general para otra corrida.
- `validate_configuration.py`: validación estructural y de presupuesto para
  un caso adaptado.

El experimento principal no utiliza las plantillas mínimas. El launcher
`run_campana_tres_brazos.sh` exige explícitamente los archivos de `reference/`
y ejecuta su propio preflight antes de inferir.

`config/experiment.env` está ignorado por Git. No guardes credenciales ni
tokens en archivos publicables.
