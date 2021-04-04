# Contribution to PyInterpolate

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

* Reporting a bug
* Discussing the current state of the code
* Submitting a fix
* Proposing new features
* Becoming a maintainer

## Where should I start?

Here, on Github! We use github to host code, to track issues and feature requests, as well as accept pull requests. We have Discord server too and it's available here. It's the fastest way to communicate with package maintainers.

### **[Discord Server](https://discord.gg/3EMuRkj)**

---

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html), So All Code Changes Happen Through Pull Requests
Pull requests are the best way to propose changes to the codebase (we use [Github Flow](https://guides.github.com/introduction/flow/index.html)). We actively welcome your pull requests:

1. Fork the repo and create your branch from `dev` or from `main` (preferably `dev`).
2. If you've added code that should be tested, add tests in the `test` package. We use Python's `unittest` package to perform testing.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the BSD 3-Clause "New" or "Revised" License
In short, when you submit code changes, your submissions are understood to be under the same [BSD 3-Clause "New" or "Revised" License] that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/szymon-datalions/pyinterpolate/issues)
We use GitHub issues to track public bugs. Report a bug by opening a new issue.

## Write bug reports with detail, background, and sample code
[This is an example](https://github.com/szymon-datalions/pyinterpolate/issues/4)

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
- Be specific!
- Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

People *love* thorough bug reports. I'm not even kidding.

## Use a PEP8 Guidelines
[PEP8 Python Guidelines](https://www.python.org/dev/peps/pep-0008/)

## License
By contributing, you agree that your contributions will be licensed under its BSD 3-Clause "New" or "Revised" License.

## References
This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md)

## Example of Contribution

1. You have an idea to speed-up computation of areal semivariance. You plan to use `multiprocessing` package for it.
2. Fork repo from `dev` branch and at the same time propose change or issue in the [project issues](https://github.com/szymon-datalions/pyinterpolate/issues). You may use two templates - one for **bug report** and other for **feature**. In this case you choose **feature**.
3. Create the new child branch from the forked `dev` branch. Name it as `dev-your-idea`. In this case `dev-areal-multiproc` is decriptive enough.
4. Code in your branch.
5. Create few unit tests in `pyinterpolate/test` directory or re-design actual tests if there is a need. For programming cases write unit tests, for mathematical and logic problems write functional tests. Use data from `sample_data` directory.
6. Multiprocessing maybe does not require new tests. But always run unittests in the `test` directory after any change in the code and check if every test has passed.
7. Run all tutorials too. Their role is not only informational. They serve as a functional test playground.
8. If everything is ok make a pull request from your forked repo.
9. And that's all! For every question use [Discord](https://discord.gg/3EMuRkj).

## Contribution by social networks

Your contribution may be other than coding itself. Questions and issues are important too. Do not be scared to write them!