# Resultados esperados del conjunto de referencia

`analyze_results.py` regenera aquí las tablas, figuras, el informe y el
manifiesto a partir de `../raw/`.

Estos archivos sirven para:

- comprobar que el análisis funciona sin GPU;
- revisar las cifras publicadas;
- conocer el formato que tendrá `analysis/` después de una corrida propia.

No son mediciones nuevas. Pueden volver a generarse en cualquier momento:

```bash
python analyze_results.py
```
