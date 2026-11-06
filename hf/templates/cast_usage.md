## Using the CAST repair

The weights here are the adapter-merged model. The repair lives in `repair/cast/` (condition
vector, layer, threshold and steering vectors, fitted with SafeTune) and is applied at
generation time. `cast_generate.py` wraps it with the same code the evaluation used
(`lift.repair.cast.make`), so install the project at the frozen tag first. Generating without
the wrapper gives the unrepaired adapter.
