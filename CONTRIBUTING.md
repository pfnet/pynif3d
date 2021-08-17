## Contributing Guidelines

### Project Pipeline

The project page provides an overall view on the backlog items, the ones being actively
developed and the ones which are blocked during development. Agile/Kanban is used as
methodology. The project-related items are grouped under the following categories:

- `Backlog`: All the newly created items will appear in this column. No developer shall
  be assigned to any of this column's items.
- `In Progress`: When a developer starts working on an item, s/he assigns
  herself/himself and moves it to this column. There shall be only one item assigned to
  one person, in order to prevent multiple WIP items at once. When a developer is
  assigned to an item, the expected completion iteration shall also be assigned.
- `WIP-Blocked`: If an item that is under development gets blocked, it shall be moved to
  this column. All the items from here will be processed with the highest priority. When
  the root cause of the block is resolved, the item shall be moved to either
  `In Progress` or the `Review` column.
- `Review`: When a developer finishes the development, test and formatting, s/he moves
  the item to this column. The item shall go back to `In Progress` or `Done` after the
  review is completed by another developer.
- `Done`: All the finished and closed items shall end up in this column.

### Labels and Definitions

Please use GitHub's issue system to submit a feature request or a bug report. The
following labels are used across the project:

#### Item Sizes

The T-shirt size definitions are used to define the size of a created item.

- `size-XS`: Items which are expected to last few hours to 1-2 days.
- `size-S`: Items which are expected to last 1-2 days to ~1 week.
- `size-M`: Items which are expected to last ~1-2 week(s).
- `size-L`: Items which are expected to last ~2-4 weeks.
- `size-XL`: Items which are expected to last more than 1 month.

As each iteration is expected to take around 1 week, large items shall be chunked into
multiple `Size-S` items.

#### Item Labels

- `enhancement`: Items which add a new feature/functionality.
- `bug`: Items which fix a bug.
- `infrastructure`: Items which aim to improve the repository infrastructure but not
  necessarily adding a new value.
- `documentation`: Items which modify the documentation of the library.

#### Item Priorities

- `low priority`: Items that will be prioritized after all the high and normal priority
  ones are addressed. Any non-value adding feature requests shall fall into this
  category.
- `normal priority`: (default) Items that will be prioritized after the high priority
  ones.
- `high priority`: Items that need to be addressed urgently. Critical bugs, important
  features shall fall into this category. Creating a high priority item may stop the
  development of the existing schedule, while causing delays and unstable outcomes.
  Hence, it shall be used very carefully.

### Submitting a Feature Request

To submit a feature request, create a new issue and use the "Feature Request" template.
Please fill out the following information:

- Detailed description (including use cases and examples, if possible). Please include
  as many details as possible.
- A concise `Definition of Done` which defines the requirements for marking an item as
  completed. Please write as many details as possible.
- Any screenshot, external link, or any other piece of information that can be helpful.

Please do not fill out the "Tasks to be completed" section when submitting a feature
request. This will be added by the developer who will be assigned to work on the item.
Also, please do not assign anyone to the issue.

By default, `enhancement`, `normal priority`, and `size-XS` labels will be assigned to
the created item. Please manually add any other labels that you deem relevant.

Lastly, assign the project `PyNIF3D` to automatically include the created item into the
backlog. Any item which doesn't have any assigned project will not be addressed.

### Reporting a Bug

To report a bug, create a new issue and use the "Bug Report" template. Please fill out
the following information:

- A detailed description of the bug. Please write as many details as possible.
- The steps that are needed to be taken in order to reproduce the bug.
- The expected behavior, if there was no bug.
- Any screenshots, if available.
- Information about your environment. Providing a Dockerfile/image is much appreciated.
- Any additional piece of information that can help us identify the problem.

Please assign the appropriate labels and the project while leaving "Assignees", "
Milestone" and "Linked Pull Requests" empty.

### Sending a Pull Request

To send a pull request, fork the PyNIF3D repository and create a branch named
`issues/XXXX` (where `XXXX` represents the ID of the issue that is addressed by the
submitted PR). Each PR has to pass the following tests:

- Unit tests
- Linter & formatting
- Code review
- Installation requirements (including any additional dependencies)

Upon creating a new PR, a developer shall be assigned for review.

## Development Guidelines

### Project Structure

The project contains several decoupled modules. The definition of the submodules are as
follows:

- `aggregation`: Functionalities related to the aggregation of a NIF model's output (
  i.e., NeRF color generation from multiple query points).
- `camera`: Functionalities related to camera operations (i.e., unprojection, pinhole
  camera model).
- `common`: Common functionalities that can be accessed throughout the library.
- `datasets`: Dataset loading functionalities.
- `encoding`: Functionalities related to the encoding of point coordinates to different
  dimensions/spaces.
- `io`: Functionalities for I/O access.
- `log`: Logging functionalities.
- `loss`: Loss-related functionalities.
- `models`: NIF models and their dependencies.
- `pipeline`: Complete algorithm pipelines. By default, a pipeline shall contain a scene
  sampler, a NIF model, and an aggregator function.
- `renderer`: Rendering-related functionalities.
- `sampling`: Sampling-related functionalities (i.e. scene, ray, pixel, plane).
- `utils`: Conversion functions and common mathematical operations.
- `vis`: Visualization-related functionalities.

### Documentation & Logging

Each class and function shall be documented to clearly explain:

- The intended process and expected behavior
- The inputs and outputs

Each class's `__ init __` function shall use the logging decorator, which logs all the
function's input and output values, if the logging level is set to `DEBUG` or lower.

### Linter & Formatting

Linter formatting is used throughout the project. Before submitting a PR, please run
linter in your environment, as follows:

```
cd $PROJECT_ROOT
bash dev/linter.sh
```

### Unit Tests

Each newly added functionality has to pass unit tests. Hence, when a new
model/pipeline/submodule is added, unit tests are expected to be included as well. Even
though there is not a standard approach on how to write unit tests, it is recommended to
consider the following cases:

- Forward test (with various optional parameters) to ensure that no exception are thrown
- Backward test, if the new module contains a custom backward pass
- Data test, if the output of the functionality is sensitive
- Tests covering edge cases