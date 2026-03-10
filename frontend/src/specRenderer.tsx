import { useEffect, useState } from "react";

import type { JsonObject, JsonValue, LegacyUiSpec, RuntimeState, UiSpecNode } from "../runtime/types";

interface SpecRendererProps {
  spec: LegacyUiSpec | null;
  state: RuntimeState | null;
  previewUrl: string;
  activeTab: string;
  onActiveTabChange: (tabId: string) => void;
  runCommand: (commandName: string, options?: { args?: JsonValue[]; kwargs?: JsonObject }) => Promise<unknown>;
  canExecuteCommand?: (commandName: string) => boolean;
}

const controlLabels: Record<string, string> = {
  "selection.prev": "Prev",
  "selection.next": "Next",
  "image.delete_current": "Delete",
  "action.run_tagger_current": "Run Tagger",
  "action.run_llm_current": "Run LLM",
  "action.run_image_process_current": "Run Image Process",
  "action.run_unmask_current": "Unmask",
  "action.run_mask_text_current": "Mask Text",
  "action.run_restore_current": "Restore",
  "task.run_tagger": "Run Tagger",
  "task.run_llm": "Run LLM",
  "task.run_image_process": "Run Image Process",
  "prompt.use_default": "Default Prompt",
  "prompt.use_custom": "Custom Prompt",
  "prompt.use_default_image_process": "Reset Prompt",
  "editor.find_replace": "Find / Replace",
  "editor.undo": "Undo",
  "editor.redo": "Redo",
};

export function SpecRenderer(props: SpecRendererProps) {
  const { spec, state, previewUrl, activeTab, onActiveTabChange, runCommand } = props;
  const [filterQuery, setFilterQuery] = useState("");
  const [indexValue, setIndexValue] = useState("1");
  const [promptText, setPromptText] = useState("");
  const [imagePromptText, setImagePromptText] = useState("");
  const [txtContent, setTxtContent] = useState("");

  useEffect(() => {
    if (!state) {
      return;
    }
    setFilterQuery(state.controls.filter_query || "");
    setIndexValue(String(state.controls.current_index || 1));
    setPromptText(state.content.prompt_text || "");
    setImagePromptText(state.content.image_process_prompt_text || "");
    setTxtContent(state.content.txt_content || "");
  }, [state]);

  if (!spec || !state) {
    return <div className="empty-state">Renderer waiting for runtime state.</div>;
  }

  return (
    <div className="spec-renderer">
      {renderNode(spec.layout, {
        state,
        previewUrl,
        activeTab,
        filterQuery,
        indexValue,
        promptText,
        imagePromptText,
        txtContent,
        setFilterQuery,
        setIndexValue,
        setPromptText,
        setImagePromptText,
        setTxtContent,
        onActiveTabChange,
        runCommand,
        canExecuteCommand: props.canExecuteCommand,
      })}
    </div>
  );
}

interface RenderContext {
  state: RuntimeState;
  previewUrl: string;
  activeTab: string;
  filterQuery: string;
  indexValue: string;
  promptText: string;
  imagePromptText: string;
  txtContent: string;
  setFilterQuery: (value: string) => void;
  setIndexValue: (value: string) => void;
  setPromptText: (value: string) => void;
  setImagePromptText: (value: string) => void;
  setTxtContent: (value: string) => void;
  onActiveTabChange: (tabId: string) => void;
  runCommand: SpecRendererProps["runCommand"];
  canExecuteCommand?: SpecRendererProps["canExecuteCommand"];
}

interface TagRenderItem {
  key: string;
  text: string;
  translation: string;
  active: boolean;
  isCharacter: boolean;
}

function renderNode(node: UiSpecNode | undefined, context: RenderContext): JSX.Element | null {
  if (!node || node.hidden) {
    return null;
  }

  switch (node.type) {
    case "split":
      return (
        <div
          className={`spec-split ${node.direction === "vertical" ? "vertical" : "horizontal"}`}
          key={String(node.id || node.title || node.type)}
        >
          {(node.children || []).map((child) => renderNode(child, context))}
        </div>
      );
    case "panel":
      if (node.id === "left_panel" || node.id === "right_panel") {
        return (
          <div className={`spec-panel spec-root-panel ${node.id === "left_panel" ? "left" : "right"}`} key={String(node.id || node.title || node.type)}>
            {(node.sections || node.children || []).map((child) => renderNode(child, context))}
          </div>
        );
      }
      return (
        <div className="spec-panel" key={String(node.id || node.title || node.type)}>
          {node.title ? <div className="spec-panel-title">{String(node.title)}</div> : null}
          {(node.sections || node.children || []).map((child) => renderNode(child, context))}
        </div>
      );
    case "toolbar":
      if (node.id === "navigation_bar") {
        const controls = node.controls || [];
        const indexControl = controls.find((control) => String(control.id || "") === "current_index");
        const totalControl = controls.find((control) => String(control.bind || "") === "selection.image_count");
        const fileControl = controls.find((control) => String(control.bind || "") === "selection.current_image_path");
        const filterInputControl = controls.find((control) => String(control.id || "") === "filter_query");
        const filterTagsControl = controls.find((control) => String(control.id || "") === "filter_tags");
        const filterTextControl = controls.find((control) => String(control.id || "") === "filter_text");
        const viewModeControl = controls.find((control) => String(control.id || "") === "view_mode");
        return (
          <div className="qt-navigation-block" key={String(node.id || node.title || node.type)}>
            <div className="qt-info-row">
              {renderNode(indexControl, context)}
              {renderNode(totalControl, context)}
              {renderNode(fileControl, context)}
            </div>
            <div className="qt-filter-row">
              {renderNode(filterInputControl, context)}
              {renderNode(filterTagsControl, context)}
              {renderNode(filterTextControl, context)}
              <button
                type="button"
                className="ghost qt-clear-filter"
                disabled={!context.state.selection.filter_active && !context.filterQuery}
                onClick={() => {
                  context.setFilterQuery("");
                  void context.runCommand("selection.clear_filter");
                }}
                title="Clear filter"
              >
                ×
              </button>
              {renderNode(viewModeControl, context)}
            </div>
          </div>
        );
      }
      return (
        <div className={`spec-toolbar ${String(node.id || "")}`} key={String(node.id || node.title || node.type)}>
          {(node.controls || []).map((control) => renderNode(control, context))}
        </div>
      );
    case "tabs":
      return (
        <div className="spec-tabs" key={String(node.id || node.title || node.type)}>
          <div className="spec-tab-strip">
            {(node.tabs || [])
              .filter((tab) => !tab.hidden)
              .map((tab) => (
                <button
                  key={String(tab.id)}
                  className={context.activeTab === tab.id ? "active" : ""}
                  onClick={() => {
                    context.onActiveTabChange(String(tab.id || ""));
                    void context.runCommand("ui.set_active_tab", { kwargs: { tab_id: String(tab.id || "") } });
                  }}
                >
                  {String(tab.title || tab.id || "tab")}
                </button>
              ))}
          </div>
          {(node.tabs || [])
            .filter((tab) => String(tab.id || "") === context.activeTab)
            .map((tab) => renderNode(tab, context))}
        </div>
      );
    case "tab":
      return (
        <div className="spec-tab" key={String(node.id || node.title || node.type)}>
          {(node.sections || []).map((section) => renderNode(section, context))}
        </div>
      );
    case "image_preview":
      return (
        <div className="spec-preview qt-checkerboard" key={String(node.id || node.title || node.type)}>
          {context.previewUrl ? (
            <img src={context.previewUrl} alt={String(resolvePath(context.state, "selection.current_image_path") || "")} />
          ) : (
            <div className="preview-empty">No preview</div>
          )}
        </div>
      );
    case "tag_flow":
      return renderTagFlow(node, context);
    case "pager":
      return (
        <div className="button-row" key={String(node.id || node.title || node.type)}>
          <button onClick={() => void context.runCommand("nl.prev_page")}>Prev Page</button>
          <button onClick={() => void context.runCommand("nl.next_page")}>Next Page</button>
          <span className="path-pill">
            {context.state.ui.nl_page_index + 1}/{context.state.ui.nl_page_count}
          </span>
        </div>
      );
    case "text_area":
      return renderTextArea(node, context);
    case "plain_text":
      return (
        <div className="editor-card text-editor-card" key={String(node.id || node.title || node.type)}>
          <div className="editor-head compact">
            <strong>{String(node.title || "Text")}</strong>
            <span className="editor-meta">{context.txtContent.length} chars</span>
          </div>
          <textarea
            rows={8}
            value={context.txtContent}
            onChange={(event) => context.setTxtContent(event.target.value)}
          />
        </div>
      );
    case "progress":
      return (
        <div className="qt-progress-row" key={String(node.id || node.title || node.type)}>
          <div className={`qt-progress-bar ${context.state.task.running ? "running" : ""}`}>
            <span className="qt-progress-fill" />
          </div>
          <span className="qt-progress-label">
            {context.state.task.running ? `Running: ${context.state.task.task_name || "task"}` : "Idle"}
          </span>
        </div>
      );
    case "button":
    case "toggle":
    case "select":
    case "label":
    case "index_input":
    case "filter_input":
      return renderControl(node, context);
    default:
      return null;
  }
}

function renderTextArea(node: UiSpecNode, context: RenderContext): JSX.Element {
  const target = node.id === "image_process_prompt" ? "image_process" : "prompt";
  const value = target === "image_process" ? context.imagePromptText : context.promptText;
  const onChange = target === "image_process" ? context.setImagePromptText : context.setPromptText;
  const commandName =
    target === "image_process" ? "content.set_image_process_prompt_text" : "content.set_prompt_text";

  return (
    <div className="editor-card" key={String(node.id || node.title || node.type)}>
      <div className="editor-head">
        <strong>{String(node.title || node.id || "Text Area")}</strong>
        <button onClick={() => void context.runCommand(commandName, { kwargs: { text: value } })}>
          Save
        </button>
      </div>
      <textarea rows={6} value={value} onChange={(event) => onChange(event.target.value)} />
    </div>
  );
}

function renderControl(node: UiSpecNode, context: RenderContext): JSX.Element | null {
  const key = String(node.id || node.command || node.type);

  if (node.type === "label") {
    const bind = String(node.bind || "");
    const value = resolvePath(context.state, bind);
    if (bind === "selection.image_count") {
      return <span className="qt-total-label" key={key}>{`/ ${String(value || 0)}`}</span>;
    }
    if (bind === "selection.current_image_path") {
      const pathText = String(value || "");
      const fileName = pathText ? pathText.split(/[/\\]/).pop() || pathText : "No image";
      return (
        <span className="qt-file-label" key={key} title={pathText}>
          {fileName}
        </span>
      );
    }
    return <span className="path-pill" key={key}>{String(value || "")}</span>;
  }

  if (node.type === "index_input") {
    return (
      <input
        key={key}
        className="spec-inline-input small qt-index-input"
        value={context.indexValue}
        onChange={(event) => context.setIndexValue(event.target.value)}
        onBlur={() => void context.runCommand("selection.jump_to_index", { kwargs: { index: Number.parseInt(context.indexValue || "1", 10) || 1 } })}
      />
    );
  }

  if (node.type === "filter_input") {
    return (
      <input
        key={key}
        className="spec-inline-input qt-filter-input"
        value={context.filterQuery}
        placeholder={String(node.label || "")}
        onChange={(event) => context.setFilterQuery(event.target.value)}
        onBlur={() =>
          void context.runCommand("selection.apply_filter", {
            kwargs: {
              query: context.filterQuery,
              use_tags: context.state.controls.filter_tags,
              use_text: context.state.controls.filter_text,
            },
          })
        }
      />
    );
  }

  if (node.type === "toggle") {
    const controlId = String(node.id || "");
    return (
      <label className="checkbox-row spec-toggle" key={key}>
        <input
          type="checkbox"
          checked={Boolean(context.state.controls[controlId] || false)}
          onChange={(event) =>
            void context.runCommand("ui.set_control_value", {
              kwargs: { control_id: controlId, value: event.target.checked },
            })
          }
        />
        <span>{String(node.label || controlId)}</span>
      </label>
    );
  }

  if (node.type === "select") {
    const controlId = String(node.id || "");
    const options = Array.isArray(node.options) && node.options.length
      ? node.options
      : [
          { label: "Original", value: 0 },
          { label: "RGB", value: 1 },
          { label: "Alpha", value: 2 },
        ];
    return (
      <select
        key={key}
        className="select-input spec-select"
        value={String(context.state.controls[controlId] ?? 0)}
        onChange={(event) =>
          void context.runCommand("ui.set_control_value", {
            kwargs: { control_id: controlId, value: Number.parseInt(event.target.value, 10) || 0 },
          })
        }
      >
        {options.map((option) => {
          const optionRecord = option as Record<string, unknown>;
          return (
            <option key={String(optionRecord.value)} value={String(optionRecord.value)}>
              {String(optionRecord.label ?? optionRecord.value)}
            </option>
          );
        })}
      </select>
    );
  }

  if (node.type === "button") {
    const commandName = resolveSpecCommand(node, context.state);
    const label = String(node.label || controlLabels[String(node.command || "")] || commandName || "Run");
    const disabled = shouldDisableButton(node, context.state, context);
    const extraClass =
      commandName === "image.delete_current"
        ? "danger"
        : commandName === "task.cancel"
          ? "ghost"
          : "";
    return (
      <button
        key={key}
        className={extraClass}
        disabled={disabled}
        onClick={() => {
          const prepared = prepareSpecCommandExecution(commandName, node, context.state);
          if (!prepared) {
            return;
          }
          void executeSpecCommand(prepared.commandName, prepared.options, context);
        }}
      >
        {label}
      </button>
    );
  }

  return null;
}

function resolvePath(target: unknown, path: string): unknown {
  if (!path) {
    return "";
  }
  return path.split(".").reduce<unknown>((current, part) => {
    if (current && typeof current === "object" && part in (current as Record<string, unknown>)) {
      return (current as Record<string, unknown>)[part];
    }
    return "";
  }, target);
}

function normalizeTagLookup(tag: string): string {
  return String(tag || "").trim().replace(/_/g, " ");
}

function normalizeTagMatch(tag: string): string {
  return String(tag || "")
    .trim()
    .replace(/, /g, "")
    .replace(/,/g, "")
    .replace(/\.$/, "")
    .trim()
    .toLowerCase();
}

function splitCsvLikeText(text: string): string[] {
  return String(text || "")
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function resolveTagTranslation(state: RuntimeState, tag: string): string {
  const translations = state.tags.translations || {};
  const exactTag = String(tag || "").trim();
  return String(
    translations[exactTag] ||
      translations[normalizeTagLookup(exactTag)] ||
      "",
  );
}

function isCharacterTag(state: RuntimeState, tag: string): boolean {
  const settings = state.settings || {};
  const value = String(tag || "").trim().toLowerCase();
  const blacklist = Array.isArray(settings.char_tag_blacklist_words)
    ? settings.char_tag_blacklist_words.map((item) => String(item || "").trim().toLowerCase()).filter(Boolean)
    : [];
  const whitelist = Array.isArray(settings.char_tag_whitelist_words)
    ? settings.char_tag_whitelist_words.map((item) => String(item || "").trim().toLowerCase()).filter(Boolean)
    : [];

  if (!blacklist.length || !value) {
    return false;
  }
  const hasBlacklist = blacklist.some((word) => value.includes(word));
  if (!hasBlacklist) {
    return false;
  }
  return !whitelist.some((word) => value.includes(word));
}

function isTagActiveInContent(state: RuntimeState, tag: string): boolean {
  const content = String(state.content.txt_content || "");
  if (!content.trim()) {
    return false;
  }
  const csvTokens = new Set(splitCsvLikeText(content).map((item) => normalizeTagMatch(item)));
  const normalizedTag = normalizeTagMatch(tag);
  if (csvTokens.has(normalizedTag)) {
    return true;
  }
  try {
    return new RegExp(`\\b${escapeRegex(String(tag || "").trim().toLowerCase())}\\b`).test(content.toLowerCase());
  } catch {
    return false;
  }
}

function resolveTagList(nodeId: unknown, state: RuntimeState): string[] {
  const key = String(nodeId || "");
  if (key === "folder_meta_tags") {
    return state.tags.folder_meta;
  }
  if (key === "custom_tags") {
    return state.tags.custom;
  }
  if (key === "tagger_tags") {
    return state.tags.tagger;
  }
  if (key === "nl_result_tags") {
    return state.tags.nl;
  }
  return [];
}

function buildTagRenderItems(nodeId: string, state: RuntimeState): TagRenderItem[] {
  const tagList =
    nodeId === "nl_result_tags"
      ? splitLatestNl(state)
      : resolveTagList(nodeId, state);

  return tagList.map((tag, index) => ({
    key: `${nodeId}-${index}-${tag}`,
    text: String(tag || ""),
    translation: nodeId === "nl_result_tags" ? "" : resolveTagTranslation(state, String(tag || "")),
    active: nodeId === "tagger_tags" ? isTagActiveInContent(state, String(tag || "")) : false,
    isCharacter: nodeId === "tagger_tags" ? isCharacterTag(state, String(tag || "")) : false,
  }));
}

function splitLatestNl(state: RuntimeState): string[] {
  const latest = String(state.content.nl_latest || "").trim();
  if (!latest) {
    return state.tags.nl || [];
  }
  const lines = latest
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length > 1) {
    return lines;
  }
  if (latest.includes(",") && !latest.includes(". ")) {
    return splitCsvLikeText(latest);
  }
  return [latest];
}

function renderTagFlow(node: UiSpecNode, context: RenderContext): JSX.Element {
  const flowId = String(node.id || "");
  const items = buildTagRenderItems(flowId, context.state);

  if (flowId === "folder_meta_tags") {
    return (
      <div className="qt-tag-section meta" key={String(node.id || node.title || node.type)}>
        {node.title ? <strong className="qt-tag-section-title">{String(node.title)}</strong> : null}
        <div className="qt-tag-input-list">
          {items.length ? (
            items.map((item) => (
              <div key={item.key} className="qt-tag-input-row">
                <div className="qt-tag-input-main">{item.text}</div>
                {item.translation ? <div className="qt-tag-input-sub">{item.translation}</div> : null}
              </div>
            ))
          ) : (
            <div className="qt-tag-empty">No tags</div>
          )}
        </div>
      </div>
    );
  }

  if (flowId === "custom_tags") {
    return (
      <div className="qt-tag-section custom" key={String(node.id || node.title || node.type)}>
        {node.title ? <strong className="qt-tag-section-title">{String(node.title)}</strong> : null}
        <div className="qt-tag-grid custom">
          {items.length ? (
            items.map((item) => (
              <div key={item.key} className="qt-tag-input-box" title={item.translation || item.text}>
                <span>{item.text}</span>
              </div>
            ))
          ) : (
            <div className="qt-tag-empty">No tags</div>
          )}
        </div>
      </div>
    );
  }

  if (flowId === "tagger_tags") {
    return (
      <div className="qt-tag-section tagger" key={String(node.id || node.title || node.type)}>
        {node.title ? <strong className="qt-tag-section-title">{String(node.title)}</strong> : null}
        <div className="qt-tag-grid tagger">
          {items.length ? (
            items.map((item) => {
              const classNames = [
                "qt-tag-card",
                item.active ? "active" : "",
                item.isCharacter && !item.active ? "character" : "",
              ]
                .filter(Boolean)
                .join(" ");
              return (
                <div key={item.key} className={classNames} title={item.text}>
                  <div className="qt-tag-name">{item.text}</div>
                  {item.translation ? <div className="qt-tag-translation">{item.translation}</div> : null}
                </div>
              );
            })
          ) : (
            <div className="qt-tag-empty">No tags</div>
          )}
        </div>
      </div>
    );
  }

  if (flowId === "nl_result_tags") {
    return (
      <div className="qt-tag-section nl" key={String(node.id || node.title || node.type)}>
        {node.title ? <strong className="qt-tag-section-title">{String(node.title)}</strong> : null}
        <div className="qt-nl-list">
          {items.length ? (
            items.map((item) => (
              <div key={item.key} className="qt-nl-row">
                {item.text}
              </div>
            ))
          ) : (
            <div className="qt-tag-empty">No content</div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="tag-section" key={String(node.id || node.title || node.type)}>
      {node.title ? <strong>{String(node.title)}</strong> : null}
      <div className="tag-cloud">
        {items.map((item) => (
          <span key={item.key} className="tag-chip">
            {item.text}
          </span>
        ))}
      </div>
    </div>
  );
}

function resolveSpecCommand(node: UiSpecNode, state: RuntimeState): string {
  const base = String(node.command || "");
  if (node.mode === "batch" && base.startsWith("task.run_")) {
    return `${base}_loaded`;
  }
  return base;
}

function buildSpecCommandOptions(node: UiSpecNode, state: RuntimeState): { args?: JsonValue[]; kwargs?: JsonObject } | undefined {
  const commandName = String(node.command || "");
  const currentImagePath = state.selection.current_image_path;
  if (node.mode === "batch" && commandName.startsWith("task.run_")) {
    const kwargs: JsonObject = {};
    if (commandName === "task.run_llm") {
      kwargs.user_prompt = state.content.prompt_text;
    }
    if (commandName === "task.run_image_process") {
      kwargs.edit_prompt = state.content.image_process_prompt_text;
    }
    return Object.keys(kwargs).length ? { kwargs } : undefined;
  }
  if (commandName === "batch.run_tagger") {
    return {
      kwargs: {
        save_to_txt: Boolean(state.controls.tagger_save_to_txt),
      },
    };
  }
  if (commandName === "batch.run_llm") {
    return {
      kwargs: {
        user_prompt: state.content.prompt_text,
        save_to_txt: Boolean(state.controls.llm_save_to_txt),
      },
    };
  }
  if (commandName === "batch.run_image_process") {
    return {
      kwargs: {
        edit_prompt: state.content.image_process_prompt_text,
      },
    };
  }
  if (commandName === "batch.run_restore") {
    return {
      kwargs: {
        confirm: true,
      },
    };
  }
  if (commandName === "action.run_llm_current") {
    return {
      kwargs: {
        user_prompt: state.content.prompt_text,
      },
    };
  }
  if (commandName === "action.run_image_process_current") {
    return {
      kwargs: {
        edit_prompt: state.content.image_process_prompt_text,
      },
    };
  }
  if (commandName.startsWith("task.run_") && currentImagePath) {
    const args: JsonValue[] = [[currentImagePath]];
    const kwargs: JsonObject = {};
    if (commandName === "task.run_llm") {
      kwargs.user_prompt = state.content.prompt_text;
    }
    if (commandName === "task.run_image_process") {
      kwargs.edit_prompt = state.content.image_process_prompt_text;
    }
    return Object.keys(kwargs).length ? { args, kwargs } : { args };
  }
  if (commandName === "image.delete_current") {
    return {
      kwargs: {
        confirm: true,
      },
    };
  }
  return undefined;
}

function shouldDisableButton(node: UiSpecNode, state: RuntimeState, context?: RenderContext): boolean {
  const commandName = String(node.command || "");
  const resolvedCommandName = resolveSpecCommand(node, state);
  if (resolvedCommandName && context?.canExecuteCommand && !context.canExecuteCommand(resolvedCommandName)) {
    return true;
  }
  if (commandName === "task.cancel") {
    return !state.task.running;
  }
  if (commandName === "action.run_mask_text_current") {
    return !state.selection.current_image_path || state.settings.mask_batch_detect_text_enabled === false;
  }
  if (commandName.startsWith("action.run_") && commandName !== "action.run_restore_current") {
    return !state.selection.current_image_path;
  }
  if (commandName === "action.run_restore_current") {
    return !state.selection.current_image_path || state.selection.has_raw_backup === false;
  }
  if (commandName.startsWith("task.run_") && node.mode !== "batch") {
    return !state.selection.current_image_path;
  }
  if (commandName.startsWith("task.run_") && node.mode === "batch") {
    return !state.selection.image_count;
  }
  if (commandName.startsWith("batch.run_")) {
    return !state.selection.image_count;
  }
  if (commandName === "image.delete_current") {
    return !state.selection.current_image_path;
  }
  if (commandName === "editor.find_replace") {
    return true;
  }
  return false;
}

function prepareSpecCommandExecution(
  commandName: string,
  node: UiSpecNode,
  state: RuntimeState,
): { commandName: string; options?: { args?: JsonValue[]; kwargs?: JsonObject } } | null {
  const baseOptions = buildSpecCommandOptions(node, state);
  const kwargs: JsonObject = { ...((baseOptions?.kwargs || {}) as JsonObject) };

  if (commandName === "image.delete_current") {
    if (!window.confirm("Move current image to no_used?")) {
      return null;
    }
    kwargs.confirm = true;
  }

  if (commandName === "batch.run_restore") {
    if (!window.confirm("Restore all loaded images from backup?")) {
      return null;
    }
    kwargs.confirm = true;
  }

  if (commandName === "batch.run_tagger" && Boolean(state.controls.tagger_save_to_txt)) {
    kwargs.delete_chars = window.confirm("Delete character tags when writing tagger output to txt?");
  }

  if (commandName === "batch.run_llm") {
    if (Boolean(state.controls.llm_save_to_txt)) {
      kwargs.delete_chars = window.confirm("Delete character tags when writing LLM output to txt?");
    }
    if (state.content.prompt_text.includes("{角色名}")) {
      if (!window.confirm("Prompt contains {角色名}. Continue batch LLM run?")) {
        return null;
      }
      kwargs.confirm_character_prompt = true;
    }
  }

  if (commandName === "action.run_llm_current") {
    if (state.content.prompt_text.includes("{tags}") && state.tags.folder_meta.length === 0 && state.tags.tagger.length === 0) {
      if (!window.confirm("Prompt uses {tags} but current image has no tag context. Continue anyway?")) {
        return null;
      }
      kwargs.confirm_missing_tags = true;
    }
  }

  return {
    commandName,
    options: Object.keys(kwargs).length ? { ...(baseOptions || {}), kwargs } : baseOptions,
  };
}

async function executeSpecCommand(
  commandName: string,
  options: { args?: JsonValue[]; kwargs?: JsonObject } | undefined,
  context: RenderContext,
): Promise<void> {
  const result = await context.runCommand(commandName, options);
  const record = result && typeof result === "object" ? (result as Record<string, unknown>) : null;
  const reason = typeof record?.reason === "string" ? record.reason : "";
  if (!reason) {
    return;
  }

  if (reason === "confirm_missing_tags_required") {
    if (!window.confirm("Prompt uses {tags} but current image has no tag context. Continue anyway?")) {
      return;
    }
    const kwargs: JsonObject = { ...((options?.kwargs || {}) as JsonObject), confirm_missing_tags: true };
    await context.runCommand(commandName, { ...(options || {}), kwargs });
    return;
  }

  if (reason === "confirm_character_prompt_required") {
    if (!window.confirm("Prompt contains {角色名}. Continue batch LLM run?")) {
      return;
    }
    const kwargs: JsonObject = { ...((options?.kwargs || {}) as JsonObject), confirm_character_prompt: true };
    await context.runCommand(commandName, { ...(options || {}), kwargs });
    return;
  }

  if (reason === "ocr_disabled") {
    window.alert("OCR is disabled in settings.");
    return;
  }

  if (reason === "no_backup") {
    window.alert("No raw backup is available for the current image.");
    return;
  }

  if (reason === "no_bg_tag_found") {
    window.alert("No loaded image contains a background tag.");
  }
}
