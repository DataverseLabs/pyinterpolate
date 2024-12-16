# Contribution to Pyinterpolate

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

* Reporting a bug
* Discussing the current state of the code
* Submitting a fix
* Proposing new features
* Becoming a maintainer

## Where should I start?

Here, on GitHub! We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.
The fastest way of communication with the core maintainers is *Discord*, and you can join the community with the link below:

### **[Discord Server](https://discord.gg/3EMuRkj)**

---

## We Use [GitHub Flow](https://guides.github.com/introduction/flow/index.html), So All Code Changes Happen Through Pull Requests
Pull requests are the best way to propose changes to the codebase (we use [Github Flow](https://guides.github.com/introduction/flow/index.html)). We actively welcome your pull requests:

1. Fork the repo and create your branch from the `main` branch.
2. If you've added code that should be tested, add tests in the `tests` package. We use the `pytest` package to perform testing.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Send the pull request to the main repository and wait for the results from automated checks. If all checks pass then a maintaner will review and accept your changes.

## Any contributions you make will be under the BSD 3-Clause "New" or "Revised" License
In short, when you submit code changes, your submissions are understood to be under the same [BSD 3-Clause "New" or "Revised" License] that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issues](https://github.com/DataverseLabs/pyinterpolate/issues)
We use GitHub issues to track public bugs. Report a bug by opening a new issue.

## Write detailed bug reports
[This is an example](https://github.com/DataverseLabs/pyinterpolate/issues/4)

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

1. You have an idea to speed-up computation of some function. You plan to use `multiprocessing`.
2. Fork repo from the `main` branch and (optionally) propose change or issue in the [project issues](https://github.com/szymon-datalions/pyinterpolate/issues). You may use two templates - one for **bug report** and other for **feature**. In this case you choose **feature**.
3. Create the new child branch from the forked `main` branch. Name it as you want, but we prefer using keywords pointing to the changes provided.
4. Code, code, code.
5. Create a few unit tests in `tests` directory, **do not alter existing tests**. For programming cases write unit tests, for mathematical and logic problems write functional tests. If you would like to use external data for tests then let us know, we must decide how will we attach the new data source into the test suite.
6. In rare occassions when changes don't require the new tests do not skip testing. Always run tests locally to ensure that you don't pass breaking changes into the repository.
7. (Optional) Run all tutorials. Their role is not only informational. They serve as a functional test playground.
8. If everything is ok make a pull request from your forked repo.
9. And that's all! If you feel overwhelmed by those steps, please let us know on [Discord](https://discord.gg/3EMuRkj).

## Other types of contributions

Your contribution may be other than coding itself.

- Questions and issues are important too. Do not be afraid to ask!
- Posting on social networks.
- Acknowledging package and citing it.