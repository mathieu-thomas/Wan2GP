# Wan2GP Plugin System Overview

## Plugin Packaging
Wan2GP treats each plugin as a regular Python package stored under the top-level `plugins/` directory. Every plugin supplies an `__init__.py` and a `plugin.py` that exposes a subclass of `shared.utils.plugins.WAN2GPPlugin`. Optional helpers, assets, and a `requirements.txt` can reside alongside those core files so that complex plugins remain self-contained.

## Lifecycle Hooks
`WAN2GPPlugin` pre-populates metadata such as `name`, `version`, and `description`, and tracks requests for components, globals, and UI insertions. Override the following hooks to integrate with the application lifecycle:

- `setup_ui()`: Declare new tabs, request Gradio components by `elem_id`, or request globals before the main interface is assembled.
- `post_ui_setup()`: Runs after the interface is assembled and gives access to resolved components so you can wire callbacks or queue `insert_after` requests.
- `on_tab_select()` / `on_tab_deselect()`: React to tab navigation events once the UI is live.

## Extending the Interface
Inside these hooks you can:

- Add custom tabs with `add_tab` to render new Gradio Blocks.
- Inject controls next to existing elements through `insert_after` for lightweight augmentations.
- Access existing components via `request_component` to read or update values, or interact with app-level state via `request_global` and `set_global`.
- Register data hooks with `register_data_hook` to inspect or mutate configuration payloads at predefined checkpoints.

## Plugin Loading and Dependency Handling
At startup `WAN2GPApplication` bootstraps the plugin manager, auto-installs missing default plugins, and reads the `enabled_plugins` list from the server configuration. Only system plugins and enabled user plugins are instantiated. After loading, the manager injects requested globals, exposes resolved components, and processes queued insertions to deliver a fully wired environment.

## Managing Plugins
The plugin manager supports discovery, installation (including editable GitHub installs and dependency resolution), update, reinstall, and uninstall operations while protecting system plugins. The Plugin Manager UI lets users toggle plugin enablement, reorder tabs, trigger maintenance actions, and browse curated community plugins via `plugins.json`. Saving changes persists the enabled list and order to the server configuration, optionally prompting an automatic restart.

## Example Use Cases
Because plugins are standard Python packages, you can build anything from UI helpers to GPU-bound pipelines. Common patterns include:

- New workflow tabs with bespoke pipelines or dashboards.
- UI augmentations that hook into existing controls, validations, or metadata flows.
- Integrations that coordinate with global services to tweak configuration or manage queues.
- Automation or analytics tasks reacting to save hooks or other data events through registered data hooks.
