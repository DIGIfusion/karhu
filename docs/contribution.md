We welcome contributions from the community! Whether you're writing an issue, fixing bugs, adding new features, or improving documentation, your help is greatly appreciated.

## How to contribute

If you find a bug or have a feature request, please open an issue on our [GitHub repository](https://github.com/DIGIfusion/karhu/issues). When reporting bugs, please provide as much detail as possible to help us reproduce the issue.

For code contributions, please fork the repository and create a new branch for your changes. Once you've made your changes, submit a pull request with a clear description of what you've done.

## Versioning

Major version bumps indicate significant changes that may include backward-incompatible changes. Minor version bumps add functionality in a backward-compatible manner. Patch version bumps are for backward-compatible bug fixes.


### Thinking in contracts

Our product is the input/output contract, the error/uncertainty semantics and the performance + reproducibility guarantees.
This is exactly like an API.

When changing the model or the code, think about how the contract changes.
There are three main types of changes:

#### 1. Compatible improvement

Same inputs, same outputs, better accuracy or calibration

Examples:
- more training data
- better loss function
- improved uncertainty calibration

-> Same major version

```
v1.2 -> v1.3
```

Downstream users should not notice anything except “it got better”.

#### 2. Contract extension

Adds optional inputs or outputs

Examples:
- new optional diagnostic output
- extra inferred quantity that downstream models may ignore

-> Minor version bump

```
v1.3 -> v1.4
```

Key rule: Old callers still work without modification.

#### 3. Contract break

Input shape changes, semantics change, or outputs redefined.

Examples:
- different normalization
- reordered outputs
- different physical meaning (e.g. growth rate vs frequency)
- uncertainty definition changes

--> Major version

```
v1.x -> v2.0
```
Downstream users must modify their code to work with the new version.

