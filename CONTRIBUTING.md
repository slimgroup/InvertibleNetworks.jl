# Contributing to InvertibleNetworks.jl

We welcome third-party contributions, and we would love you to become an active contributor!

Software contributions are made via GitHub pull requests. If you are planning a large contribution, we encourage you to engage with us frequently to ensure that your effort is well-directed. See below for more details.

This repository is distributed under the MIT License, https://github.com/slimgroup/InvertibleNetworks.jl/blob/master/LICENSE. The act of submitting a pull request or patch (with or without an explicit Signed-off-by tag) will be understood as an affirmation of the following:

 Developer's Certificate of Origin 1.1

 By making a contribution to this project, I certify that:

 (a) The contribution was created in whole or in part by me and I
   have the right to submit it under the open source license
   indicated in the file; or

 (b) The contribution is based upon previous work that, to the best
   of my knowledge, is covered under an appropriate open source
   license and I have the right under that license to submit that
   work with modifications, whether created in whole or in part
   by me, under the same open source license (unless I am
   permitted to submit under a different license), as indicated
   in the file; or

 (c) The contribution was provided directly to me by some other
   person who certified (a), (b) or (c) and I have not modified
   it.

 (d) I understand and agree that this project and the contribution
   are public and that a record of the contribution (including all
   personal information I submit with it, including my sign-off) is
   maintained indefinitely and may be redistributed consistent with
   this project or the open source license(s) involved.

### Reporting issues

There are several options:
* Talk to us. The current main developer is Rafael Orozco [rorozco@gatech.edu]
* File an issue or start a discussion on the repository.

### Making changes

First of all, read our [code of conduct](https://github.com/slimgroup/.github/blob/master/CODE_OF_CONDUCT.md) and make sure you agree with it.

The protocol to propose a patch is:
* [Recommended, but not compulsory] Talk to us about what you're trying to do. There is a great chance we can support you.
* As soon as you know what you need to do, [fork](https://help.github.com/articles/fork-a-repo/) the repository.
* Create a branch with a suitable name.
* Write code following the guidelines below. Commit your changes as small logical units.
* Write meaningful commit messages.
* Write tests to convince us and yourself that what you've done works as expected. Commit them.
* Run **the entire test suite**, including the new tests, to make sure that you haven't accidentally broken anything else.
* Push everything to your fork.
* Submit a Pull Request on our repository.
* Wait for us to provide feedback. This may require a few iterations.

Tip, especially for newcomers: prefer short, self-contained Pull Requests over lengthy, impenetrable, and thus difficult to review, ones.

### Coding guidelines

Some coding rules are "enforced" (and automatically checked by our Continuous Integration systems), some are "strongly recommended", others are "optional" but welcome.

* We _strongly recommend_ to document any new module, type, method, ... with [julia documentation](https://docs.julialang.org/en/v1/manual/documentation/).
* We _strongly recommend_ to follow standard Julia coding guidelines:
  - Use camel caps for struct names, e.g. ``struct FooBar``.
  - Method names must start with a small letter; use underscores to separate words, e.g. ``function my_meth_...``.
  - Variable names should be explicative (We prefer "long and clear" over "short but unclear").
  - Comment your code, and do not be afraid of being verbose.
* We _like_ that blank lines are used to logically split blocks of code implementing different (possibly sequential) tasks.

### Adding tutorials or examples

We always look forward to extending our [suite of tutorials and examples](https://github.com/slimgroup/SLIMTutorials) with new Jupyter Notebooks. Even something completely new, such as a new series of tutorials showing your work with our softwares, would be a great addition.
