Rules for interactions:
* Ask questions whenever you need clarification, or there are multiple ways to intepret my instructions. Do not just simply make a decision on your own.
* In plan mode, do NOT write plan files unless instructed. I just want to discuss.

Virtual env:
Use: `source .venv/bin/activate` (it is a uv pip)

Rules for code:
* Prioritize writing easy to read code. That means modularity, leveraging software packages (pandas, numpy, torch) when appropriate to minimize custom code.
* NO silent failures. Use assert statements when needed. Do not use try catches.
* No stupid variable assignments like `a = b`. This happens a lot in the code and is just leading to code bloat.
* Write modular code. Avoid writing code that is basically repeat of another. To minimize repeated code, write functions that can be shared.
* All experiments are organized via scons and nestly. All input and output paths should be specified as arguments to python/R scripts as well as the sconscript files.
* Code should be reproducible with random seeds
* For random choices, use a seeded numpy random number generator (`np.random.default_rng(seed)`). Never reimplement randomization (e.g., hashing strings with `hashlib` to pick an index); it is much harder to read.
* Package imports and globals are always at the top of the python file
* All code will be organized via scons.
* never use --force for github
* NEVER run `rm -rf` (or any recursive force delete). To rebuild outputs, change the command or sources and let scons rebuild the targets; ask the user if something must be deleted.
* Fail loudly while the software is being built: LLM responses that fail validation, crashing/timing-out/invalid LLM-generated code, and other errors all halt the run, with logs that make the failure easy to trace. Only downgrade an error to a logged warning after we explicitly decide it's acceptable.
* Anything specific to an experiment (SAP fields, outcome columns, prompts, response schemas) is defined in that experiment's config and prompt files, not in package code. Others should be able to run new experiments without editing `bayes_multiverse/`.
* While building, start small: one seed and few iterations/concepts.
* Only write tests that check something non-trivial (e.g., math against ground truth), or no tests.
