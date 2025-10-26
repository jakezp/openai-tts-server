---
id: troubleshooting
aliases: []
tags: []
---

#### `uvx` command not found

You are either running the command from within the Docker container or the `uvx` command is not installed. This likely means you are trying to run the command from within the Docker container, but the `uvx` command line tool is not installed in the container. You have two options to resolve this issue:

- (**Recommended**) Run the `uvx` command from the host machine instead of the Docker container. If you don't have the `uvx` command line tool installed on your host machine, you can install it by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
- You could also install the `uvx` command line tool inside the Docker container. NOTE: any changes you make to the container will be lost when the container is stopped or removed. To install the `uvx` command line tool inside the Docker container, you can should use the same installation instructions as you would for the host machine.
