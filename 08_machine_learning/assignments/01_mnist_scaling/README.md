# Assignment 01: MNIST Scaling on CPU and GPU

Implement and measure a handwritten-digit classifier using PyTorch.

Minimum requirements:

1. command-line arguments for `epochs`, `batch-size`, and `device`
2. training loss output for each epoch
3. test accuracy
4. timing output for total training time
5. one comparison between CPU and GPU if a GPU is available

Recommended study:

- batch size `64`
- batch size `128`
- batch size `256`
- batch size `512`

Questions to answer:

1. How does batch size affect runtime?
2. How does batch size affect accuracy?
3. Does GPU acceleration help at all tested batch sizes?
4. What part of the workflow is still not fully explained by raw compute speed?

Deliverables:

- source code
- raw timing logs
- one short report with plots or tables

Notes:

- do not fabricate results
- state clearly if your machine does not provide a usable GPU
