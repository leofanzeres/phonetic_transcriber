# phonetic_transcriber

The phonetic transcriber is a tool for converting the letter representation of a word to a phoneme representation. The tool uses Recurrent Neural Networks (RNNs) and currently provides two features:
* train models: ```actions/train.py)```
* evaluate models: ```actions/evaluation.py)```

Since we use relative paths for modules, you may need to set PYTHONPATH system variable:
```
export PYTHONPATH="${PYTHONPATH}:/path/to/package/"
```

### Requirements (tested versions)
pytest 7.2.1
torch 1.13.1
