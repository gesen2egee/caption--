import { useEffect, useRef, useState, type ReactNode } from "react";

import { CaptionRuntimeClient, CaptionRuntimeError } from "../runtime/client";
import { SpecRenderer } from "./specRenderer";
import type {
  AgentManifest,
  CommandMetadata,
  CommandsSummary,
  JsonObject,
  JsonValue,
  LegacyUiSpec,
  RuntimeCapabilities,
  RuntimeEvent,
  RuntimeState,
  SettingField,
  SettingsSchema,
  UiSpecNode,
  WorkersSummary,
} from "../runtime/types";

const runtimeBaseUrl =
  import.meta.env.VITE_CAPTION_RUNTIME_URL || (import.meta.env.DEV ? "/api" : "");
const client = new CaptionRuntimeClient(runtimeBaseUrl);
const CONTROL_EVENT_PROFILE = "control";

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string" || !result) {
        reject(new Error("failed to read file"));
        return;
      }
      resolve(result);
    };
    reader.onerror = () => reject(reader.error || new Error("failed to read file"));
    reader.readAsDataURL(file);
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function applyRuntimeStateEvent(current: RuntimeState | null, event: RuntimeEvent): RuntimeState | null {
  const payload = isRecord(event.payload) ? event.payload : {};
  if (payload.full_state === true && isRecord(payload.state)) {
    return payload.state as unknown as RuntimeState;
  }

  if (typeof payload.section === "string" && payload.section && "section_state" in payload) {
    if (!current) {
      return current;
    }
    return {
      ...current,
      [payload.section]: payload.section_state as RuntimeState[keyof RuntimeState],
    };
  }

  if (typeof payload.key === "string" && payload.key) {
    if (!current) {
      return current;
    }
    return {
      ...current,
      [payload.key]: payload.value as RuntimeState[keyof RuntimeState],
    };
  }

  return current;
}

function applySettingsSchemaValues(
  current: SettingsSchema | null,
  nextSettings: unknown,
): SettingsSchema | null {
  if (!current || !isRecord(nextSettings)) {
    return current;
  }
  return {
    ...current,
    fields: current.fields.map((field) => ({
      ...field,
      value: (nextSettings[field.key] as JsonValue | undefined) ?? field.value,
    })),
  };
}

function hasUiSpecOverride(value: unknown): boolean {
  return isRecord(value) && Object.keys(value).length > 0;
}

function shouldRefreshPreviewForEvent(event: RuntimeEvent, currentImagePath: string): boolean {
  if (!currentImagePath) {
    return false;
  }
  if (event.name === "image.stroke_erased") {
    return true;
  }
  if (event.name === "task.image_done") {
    const payload = isRecord(event.payload) ? event.payload : {};
    return String(payload.image_path || "") === currentImagePath;
  }
  return false;
}

function commandRiskClass(metadata?: CommandMetadata | null): string {
  if (metadata?.risk_level === "high") {
    return "risk-high";
  }
  if (metadata?.risk_level === "medium") {
    return "risk-medium";
  }
  return "risk-low";
}

function commandGuardrailLabels(metadata?: CommandMetadata | null): string[] {
  if (!metadata) {
    return [];
  }
  const labels: string[] = [];
  if (metadata.category_label) {
    labels.push(metadata.category_label);
  }
  if (metadata.read_only) {
    labels.push("read-only");
  }
  if (metadata.long_running) {
    labels.push("long-running");
  }
  if (metadata.writes_files) {
    labels.push("writes-files");
  }
  if (metadata.destructive) {
    labels.push("destructive");
  }
  if (metadata.requires_confirmation) {
    labels.push("confirm");
  }
  if (metadata.mutates_settings) {
    labels.push("settings");
  }
  if (metadata.mutates_ui) {
    labels.push("ui-mutation");
  }
  if (metadata.mutates_runtime) {
    labels.push("runtime");
  }
  return labels;
}

function describeError(error: unknown): string {
  if (error instanceof CaptionRuntimeError) {
    const code = error.info?.code ? `[${error.info.code}] ` : "";
    const source = error.info?.source ? ` (${error.info.source})` : "";
    return `${code}${error.message}${source}`;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tagName = target.tagName.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select" || target.isContentEditable;
}

const WEB_TEXT = {
  zh_tw: {
    title: "Caption Tool",
    runtime: "Web Runtime",
    folder: "資料夾",
    load: "載入",
    shutdown: "關閉 Runtime",
    file: "檔案",
    openDir: "開啟目錄",
    refresh: "重新整理",
    settings: "設定",
    edit: "編輯",
    findReplace: "尋找/取代",
    addTag: "新增標籤",
    tools: "工具",
    unmask: "單圖去背景",
    maskText: "單圖去文字",
    restore: "放回原圖",
    batchUnmask: "Batch 去背景",
    batchMaskText: "Batch 去文字",
    batchRestore: "Batch 放回原圖",
    strokeEraser: "手繪橡皮擦",
    view: "檢視",
    advanced: "進階面板",
    advancedPanels: "進階 Runtime 面板",
    close: "關閉",
    apply: "套用",
    allImages: "全部圖片",
    caseSensitive: "區分大小寫",
    regex: "正規表示式",
    uploadMaskHint: "上傳白色=擦除、黑色=保留的遮罩圖",
    mode: "模式",
    stream: "事件流",
    bridge: "Bridge",
    task: "任務",
    workers: "Workers",
    noSelection: "未選取項目",
    latestResult: "最近工具結果",
    settingsGroup: "設定群組",
  },
  en: {
    title: "Caption Tool",
    runtime: "Web Runtime",
    folder: "Folder",
    load: "Load",
    shutdown: "Shutdown Runtime",
    file: "File",
    openDir: "Open Directory",
    refresh: "Refresh",
    settings: "Settings",
    edit: "Edit",
    findReplace: "Find / Replace",
    addTag: "Add Tag",
    tools: "Tools",
    unmask: "Unmask",
    maskText: "Mask Text",
    restore: "Restore",
    batchUnmask: "Batch Unmask",
    batchMaskText: "Batch Mask Text",
    batchRestore: "Batch Restore",
    strokeEraser: "Stroke Eraser",
    view: "View",
    advanced: "Advanced Panels",
    advancedPanels: "Advanced Runtime Panels",
    close: "Close",
    apply: "Apply",
    allImages: "All images",
    caseSensitive: "Case sensitive",
    regex: "Regex",
    uploadMaskHint: "Upload a white=erase, black=keep mask image",
    mode: "Mode",
    stream: "Stream",
    bridge: "Bridge",
    task: "Task",
    workers: "Workers",
    noSelection: "No selection",
    latestResult: "Last Tool Result",
    settingsGroup: "Settings Group",
  },
} as const;

function getWebText(language?: string) {
  if (language === "en") {
    return WEB_TEXT.en;
  }
  return WEB_TEXT.zh_tw;
}

type DesktopMenuId = "" | "file" | "edit" | "tools" | "view";

interface DesktopMenuItem {
  label: string;
  shortcut?: string;
  disabled?: boolean;
  danger?: boolean;
  onSelect: () => void;
}

interface DesktopMenuGroup {
  id: Exclude<DesktopMenuId, "">;
  label: string;
  items: DesktopMenuItem[];
}

export default function App() {
  const [summary, setSummary] = useState<CommandsSummary | null>(null);
  const [runtimeState, setRuntimeState] = useState<RuntimeState | null>(null);
  const [uiSpec, setUiSpec] = useState<LegacyUiSpec | null>(null);
  const [uiSpecOverride, setUiSpecOverride] = useState<Record<string, unknown>>({});
  const [capabilities, setCapabilities] = useState<RuntimeCapabilities | null>(null);
  const [agentManifest, setAgentManifest] = useState<AgentManifest | null>(null);
  const [agentSurfaceMode, setAgentSurfaceMode] = useState("development");
  const [workers, setWorkers] = useState<WorkersSummary | null>(null);
  const [settingsSchema, setSettingsSchema] = useState<SettingsSchema | null>(null);
  const [events, setEvents] = useState<RuntimeEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busyCommand, setBusyCommand] = useState<string | null>(null);
  const [previewNonce, setPreviewNonce] = useState(Date.now());
  const [activeTab, setActiveTab] = useState("tags_tab");
  const [rootDirDraft, setRootDirDraft] = useState("");
  const [rootDirDirty, setRootDirDirty] = useState(false);
  const [filterDraft, setFilterDraft] = useState("");
  const [filterDirty, setFilterDirty] = useState(false);
  const [uiSpecDraft, setUiSpecDraft] = useState("{}");
  const [uiSpecDirty, setUiSpecDirty] = useState(false);
  const [selectedSettingsGroup, setSelectedSettingsGroup] = useState("");
  const [settingDrafts, setSettingDrafts] = useState<Record<string, unknown>>({});
  const [customTagDraft, setCustomTagDraft] = useState("");
  const [findTextDraft, setFindTextDraft] = useState("");
  const [replaceTextDraft, setReplaceTextDraft] = useState("");
  const [findScopeAll, setFindScopeAll] = useState(false);
  const [findCaseSensitive, setFindCaseSensitive] = useState(false);
  const [findRegex, setFindRegex] = useState(false);
  const [toolResult, setToolResult] = useState<string>("");
  const [strokeMaskName, setStrokeMaskName] = useState("");
  const [strokeMaskDataUrl, setStrokeMaskDataUrl] = useState("");
  const [eventStreamStatus, setEventStreamStatus] = useState<"connecting" | "live" | "fallback">("connecting");
  const [quickDialog, setQuickDialog] = useState<"" | "settings" | "findReplace" | "stroke" | "customTag">("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [activeMenu, setActiveMenu] = useState<DesktopMenuId>("");
  const refreshTimerRef = useRef<number | null>(null);
  const currentImagePathRef = useRef("");
  const agentSurfaceModeRef = useRef(agentSurfaceMode);
  const rootDirInputRef = useRef<HTMLInputElement | null>(null);
  const menuBarRef = useRef<HTMLElement | null>(null);

  const refreshRuntime = async () => {
    const activeMode = agentSurfaceModeRef.current;
    try {
      const [nextSummary, nextState, nextUiSpec, nextUiSpecOverride, nextEvents, nextCapabilities, nextAgentManifest, nextWorkers, nextSettingsSchema] = await Promise.all([
        client.getCommands(activeMode),
        client.getState(),
        client.getUiSpec(),
        client.getUiSpecOverride(),
        client.getEvents(120, { profile: CONTROL_EVENT_PROFILE }),
        client.getCapabilities(activeMode),
        client.getAgentManifest(activeMode),
        client.getWorkers(),
        client.getSettingsSchema(),
      ]);
      setSummary(nextSummary);
      setRuntimeState(nextState);
      setUiSpec(nextUiSpec);
      setUiSpecOverride(nextUiSpecOverride);
      setEvents(nextEvents);
      setCapabilities(nextCapabilities);
      setAgentManifest(nextAgentManifest);
      setWorkers(nextWorkers);
      setSettingsSchema(nextSettingsSchema);
      setError(null);
    } catch (nextError) {
      setError(describeError(nextError));
    }
  };

  const scheduleRefreshRuntime = () => {
    if (refreshTimerRef.current !== null) {
      return;
    }
    refreshTimerRef.current = window.setTimeout(() => {
      refreshTimerRef.current = null;
      void refreshRuntime();
    }, 140);
  };

  useEffect(() => {
    void refreshRuntime();
    const eventStream = client.subscribeEvents(
      (event) => {
        if (event.name === "stream.ready" || event.name.startsWith("bridge.")) {
          setEvents((current) => {
            const next = [...current, event];
            return next.slice(-120);
          });
        } else if (
          event.name.startsWith("command.") ||
          event.name.startsWith("task.") ||
          event.name.startsWith("worker.") ||
          event.name.startsWith("settings.") ||
          event.name.startsWith("ui.spec.") ||
          event.name.startsWith("image.") ||
          event.name.startsWith("editor.") ||
          event.name.startsWith("tags.custom.")
        ) {
          setEvents((current) => {
            const next = [...current, event];
            return next.slice(-120);
          });
        }

        if (shouldRefreshPreviewForEvent(event, currentImagePathRef.current)) {
          setPreviewNonce(Date.now());
        }

        let handled = false;
        if (event.name === "ui.spec.updated" && isRecord(event.payload)) {
          if (isRecord(event.payload.spec)) {
            setUiSpec(event.payload.spec as unknown as LegacyUiSpec);
            handled = true;
          }
          if (isRecord(event.payload.override)) {
            setUiSpecOverride(event.payload.override);
            setCapabilities((current) =>
              current
                ? {
                    ...current,
                    ui_spec_override_active: hasUiSpecOverride(event.payload.override),
                  }
                : current,
            );
            handled = true;
          }
        } else if (
          event.name === "state.updated" ||
          event.name === "state.replaced" ||
          event.name.endsWith(".updated")
        ) {
          setRuntimeState((current) => applyRuntimeStateEvent(current, event));
          handled = true;

          if (event.name === "settings.updated" && isRecord(event.payload) && "section_state" in event.payload) {
            setSettingsSchema((current) => applySettingsSchemaValues(current, event.payload.section_state));
          }
        } else if (event.name === "worker.scan.finished" && isRecord(event.payload)) {
          setWorkers(event.payload as unknown as WorkersSummary);
          setCapabilities((current) =>
            current
              ? {
                  ...current,
                  workers: event.payload as unknown as WorkersSummary,
                }
              : current,
          );
          handled = true;
        } else if (
          (event.name === "bridge.started" || event.name === "bridge.stopped") &&
          isRecord(event.payload)
        ) {
          setSummary((current) =>
            current
              ? {
                  ...current,
                  bridge: event.payload as unknown as CommandsSummary["bridge"],
                }
              : current,
          );
          setCapabilities((current) =>
            current
              ? {
                  ...current,
                  bridge: event.payload as unknown as RuntimeCapabilities["bridge"],
                }
              : current,
          );
          handled = true;
        } else if (
          event.name.startsWith("command.") ||
          event.name.startsWith("task.") ||
          event.name === "image.deleted" ||
          event.name === "image.stroke_erased" ||
          event.name === "editor.find_replace.completed" ||
          event.name === "tags.custom.added" ||
          event.name === "worker.scan.started"
        ) {
          handled = true;
        }

        if (event.name !== "stream.ready" && !handled) {
          scheduleRefreshRuntime();
        }
      },
      (status) => {
        if (status === "live") {
          setEventStreamStatus("live");
          return;
        }
        if (status === "error" || status === "closed") {
          setEventStreamStatus("fallback");
        }
      },
      {},
    );
    if (eventStream === null) {
      setEventStreamStatus("fallback");
    }
    const timer = window.setInterval(() => {
      void refreshRuntime();
    }, 12000);
    return () => {
      window.clearInterval(timer);
      if (refreshTimerRef.current !== null) {
        window.clearTimeout(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
      eventStream?.close();
    };
  }, []);

  useEffect(() => {
    if (!runtimeState) {
      return;
    }
    if (!rootDirDirty) {
      setRootDirDraft(runtimeState.selection.root_dir_path || "");
    }
    if (!filterDirty) {
      setFilterDraft(runtimeState.selection.filter_query || "");
    }
    if (!uiSpecDirty) {
      setUiSpecDraft(JSON.stringify(uiSpecOverride, null, 2));
    }
    setActiveTab(resolveTabId(uiSpec, runtimeState.ui.current_tab_index));
    setPreviewNonce(Date.now());
  }, [
    runtimeState,
    uiSpec,
    uiSpecOverride,
    rootDirDirty,
    filterDirty,
    uiSpecDirty,
  ]);

  useEffect(() => {
    currentImagePathRef.current = runtimeState?.selection.current_image_path || "";
  }, [runtimeState?.selection.current_image_path]);

  useEffect(() => {
    agentSurfaceModeRef.current = agentSurfaceMode;
  }, [agentSurfaceMode]);

  useEffect(() => {
    setPreviewNonce(Date.now());
  }, [runtimeState?.selection.current_image_path]);

  useEffect(() => {
    if (!selectedSettingsGroup && settingsSchema?.groups.length) {
      setSelectedSettingsGroup(settingsSchema.groups[0]);
    }
  }, [selectedSettingsGroup, settingsSchema]);

  useEffect(() => {
    void Promise.all([
      client.getCommands(agentSurfaceMode),
      client.getCapabilities(agentSurfaceMode),
      client.getAgentManifest(agentSurfaceMode),
    ])
      .then(([nextSummary, nextCapabilities, manifest]) => {
        setSummary(nextSummary);
        setCapabilities(nextCapabilities);
        setAgentManifest(manifest);
        setError(null);
      })
      .catch((nextError) => {
        setError(describeError(nextError));
      });
  }, [agentSurfaceMode]);

  const runCommand = async (
    commandName: string,
    options: { args?: JsonValue[]; kwargs?: JsonObject } = {},
  ) => {
    if (!canExecuteCommand(commandName)) {
      setError(`Command '${commandName}' is not available in ${agentSurfaceMode} mode.`);
      return null;
    }
    setBusyCommand(commandName);
    try {
      const response = await client.executeCommand(commandName, {
        args: options.args,
        kwargs: options.kwargs,
      }, agentSurfaceMode);
      if (!response.ok) {
        throw new CaptionRuntimeError(
          response.error || `Command failed: ${commandName}`,
          response.error_info,
        );
      }
      await refreshRuntime();
      return response.result;
    } catch (nextError) {
      setError(describeError(nextError));
      return null;
    } finally {
      setBusyCommand(null);
    }
  };

  const currentImagePath = runtimeState?.selection.current_image_path || "";
  const previewUrl = currentImagePath ? client.getCurrentPreviewUrl(previewNonce) : "";
  const visibleEvents = events.slice(-8).reverse();
  const workerCards = Object.entries(workers?.categories || {}).filter(([, items]) => items.length);
  const workerServiceCount = workers?.service_manager?.service_count || 0;
  const workerRuntimeMode = workers?.runtime_mode || capabilities?.worker_runtime_mode || "inprocess";
  const reloadableRuntimeServices = Object.entries(capabilities?.runtime_reloadable_services || {});
  const runtimeServiceWatch = capabilities?.runtime_service_watch_status;
  const settingsPreview = (settingsSchema?.fields || []).slice(0, 12);
  const editableSettings = (settingsSchema?.fields || []).filter(
    (field) => field.group === (selectedSettingsGroup || settingsSchema?.groups[0]),
  );
  const capabilityCommandMetadata = capabilities?.command_metadata || {};
  const agentCommandMetadata = agentManifest?.command_metadata || {};
  const restrictedAgentCommandMetadata = agentManifest?.restricted_command_metadata || {};
  const allowedCommandSet = new Set(capabilities?.commands || []);
  const destructiveCapabilityCount = Object.values(capabilityCommandMetadata).filter(
    (metadata) => metadata?.destructive,
  ).length;
  const recommendedAgentCommands = (agentManifest?.recommended_commands || []).slice(0, 8);
  const restrictedAgentCommands = (agentManifest?.restricted_commands || []).slice(0, 8);
  const canExecuteCommand = (commandName: string): boolean => allowedCommandSet.has(commandName);
  const currentFileName = currentImagePath ? currentImagePath.split(/[/\\]/).pop() || currentImagePath : "";
  const currentRootDir = runtimeState?.selection.root_dir_path || rootDirDraft;
  const uiLanguage = typeof runtimeState?.settings.ui_language === "string" ? runtimeState.settings.ui_language : undefined;
  const webText = getWebText(uiLanguage);

  const focusRootDirInput = () => {
    rootDirInputRef.current?.focus();
    rootDirInputRef.current?.select();
  };

  const runDirectoryRefresh = () => {
    if (!currentRootDir) {
      focusRootDirInput();
      return;
    }
    void runCommand("selection.set_root_dir", {
      kwargs: { dir_path: currentRootDir },
    }).then(() => setRootDirDirty(false));
  };

  const runBatchRestore = () => {
    if (!window.confirm("Restore all loaded images from backup?")) {
      return;
    }
    void runCommand("batch.run_restore", {
      kwargs: { confirm: true },
    }).then((result) => setToolResult(JSON.stringify(result ?? {}, null, 2)));
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && activeMenu) {
        event.preventDefault();
        setActiveMenu("");
        return;
      }
      if (event.key === "Escape" && quickDialog) {
        event.preventDefault();
        setQuickDialog("");
        return;
      }

      if (isEditableTarget(event.target)) {
        return;
      }

      const isMeta = event.ctrlKey || event.metaKey;

      if (event.key === "F5") {
        event.preventDefault();
        runDirectoryRefresh();
        return;
      }

      if (isMeta && event.key.toLowerCase() === "o") {
        event.preventDefault();
        focusRootDirInput();
        return;
      }

      if (isMeta && event.key.toLowerCase() === "f" && canExecuteCommand("editor.find_replace")) {
        event.preventDefault();
        setQuickDialog("findReplace");
        return;
      }

      if (isMeta && event.key === "," && canExecuteCommand("settings.update")) {
        event.preventDefault();
        setQuickDialog("settings");
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeMenu, quickDialog, currentRootDir, capabilities, rootDirDraft, agentSurfaceMode, settingsSchema, canExecuteCommand, runCommand]);

  useEffect(() => {
    const onPointerDown = (event: MouseEvent) => {
      if (!menuBarRef.current) {
        return;
      }
      if (!menuBarRef.current.contains(event.target as Node)) {
        setActiveMenu("");
      }
    };
    window.addEventListener("mousedown", onPointerDown);
    return () => window.removeEventListener("mousedown", onPointerDown);
  }, []);

  const menuGroups: DesktopMenuGroup[] = [
    {
      id: "file",
      label: webText.file,
      items: [
        {
          label: webText.openDir,
          shortcut: "Ctrl+O",
          onSelect: focusRootDirInput,
        },
        {
          label: webText.refresh,
          shortcut: "F5",
          onSelect: runDirectoryRefresh,
        },
        {
          label: webText.settings,
          shortcut: "Ctrl+,",
          disabled: !canExecuteCommand("settings.update"),
          onSelect: () => setQuickDialog("settings"),
        },
      ],
    },
    {
      id: "edit",
      label: webText.edit,
      items: [
        {
          label: webText.findReplace,
          shortcut: "Ctrl+F",
          disabled: !canExecuteCommand("editor.find_replace"),
          onSelect: () => setQuickDialog("findReplace"),
        },
        {
          label: webText.addTag,
          disabled: !currentImagePath || !canExecuteCommand("tags.add_custom"),
          onSelect: () => setQuickDialog("customTag"),
        },
      ],
    },
    {
      id: "tools",
      label: webText.tools,
      items: [
        {
          label: webText.unmask,
          disabled: !currentImagePath || !canExecuteCommand("action.run_unmask_current"),
          onSelect: () => {
            void runCommand("action.run_unmask_current");
          },
        },
        {
          label: webText.maskText,
          disabled: !currentImagePath || !canExecuteCommand("action.run_mask_text_current"),
          onSelect: () => {
            void runCommand("action.run_mask_text_current");
          },
        },
        {
          label: webText.restore,
          disabled: !currentImagePath || !canExecuteCommand("action.run_restore_current"),
          onSelect: () => {
            void runCommand("action.run_restore_current");
          },
        },
        {
          label: webText.batchUnmask,
          disabled: !runtimeState?.selection.image_count || !canExecuteCommand("batch.run_unmask"),
          onSelect: () => {
            void runCommand("batch.run_unmask");
          },
        },
        {
          label: webText.batchMaskText,
          disabled: !runtimeState?.selection.image_count || !canExecuteCommand("batch.run_mask_text"),
          onSelect: () => {
            void runCommand("batch.run_mask_text");
          },
        },
        {
          label: webText.batchRestore,
          disabled: !runtimeState?.selection.image_count || !canExecuteCommand("batch.run_restore"),
          onSelect: runBatchRestore,
        },
        {
          label: webText.strokeEraser,
          disabled: !currentImagePath || !canExecuteCommand("action.run_stroke_eraser_current"),
          onSelect: () => setQuickDialog("stroke"),
        },
      ],
    },
    {
      id: "view",
      label: webText.view,
      items: [
        {
          label: webText.advanced,
          onSelect: () => setAdvancedOpen((current) => !current),
        },
      ],
    },
  ];

  return (
    <div className="qt-web-app">
      <header className="qt-toolbar">
        <div className="qt-toolbar-title">
          <strong>{webText.title}</strong>
          <span>{webText.runtime}</span>
        </div>
        <div className="qt-toolbar-row">
          <label htmlFor="root-dir">{webText.folder}</label>
          <input
            id="root-dir"
            ref={rootDirInputRef}
            value={rootDirDraft}
            onChange={(event) => {
              setRootDirDraft(event.target.value);
              setRootDirDirty(true);
            }}
            placeholder="E:\\images\\set"
          />
          <button
            onClick={() =>
              void runCommand("selection.set_root_dir", {
                kwargs: { dir_path: rootDirDraft },
              }).then(() => setRootDirDirty(false))
            }
          >
            {webText.load}
          </button>
          <button
            className="ghost"
            disabled={!canExecuteCommand("app.shutdown")}
            onClick={() => {
              if (!window.confirm("Shutdown the Caption runtime host?")) {
                return;
              }
              void runCommand("app.shutdown", {
                kwargs: { confirm: true },
              });
            }}
          >
            {webText.shutdown}
          </button>
        </div>
      </header>

      <DesktopMenuBar
        containerRef={menuBarRef}
        groups={menuGroups}
        activeMenu={activeMenu}
        onActiveMenuChange={setActiveMenu}
      />

      {error ? <div className="banner error">{error}</div> : null}
      {busyCommand ? <div className="banner working">Executing `{busyCommand}`</div> : null}

      <main className="qt-main">
        <section className="qt-main-panel">
            <SpecRenderer
              spec={uiSpec}
              state={runtimeState}
              previewUrl={previewUrl}
              activeTab={activeTab}
              onActiveTabChange={setActiveTab}
              runCommand={runCommand}
              canExecuteCommand={canExecuteCommand}
            />
        </section>
      </main>

      <footer className="qt-statusbar">
        <span className="qt-status-item">{webText.bridge}: {summary?.bridge.enabled ? "online" : "offline"}</span>
        <span className="qt-status-item">{webText.task}: {runtimeState?.task.running ? runtimeState.task.task_name || "running" : "idle"}</span>
        <span className="qt-status-item">{webText.workers}: {workerCards.length}</span>
        <span className="qt-status-item">{webText.stream}: {eventStreamStatus}</span>
        <span className="qt-status-item">{webText.mode}: {agentSurfaceMode}</span>
        <span className="qt-status-item qt-status-path" title={currentImagePath || runtimeState?.selection.root_dir_path || ""}>
          {currentFileName || runtimeState?.selection.root_dir_path || webText.noSelection}
        </span>
      </footer>

      <details className="qt-advanced" open={advancedOpen} onToggle={(event) => setAdvancedOpen((event.currentTarget as HTMLDetailsElement).open)}>
        <summary>{webText.advancedPanels}</summary>
        <main className="workspace">
          <section className="column">
          <Panel title="Workers" subtitle={`${workers?.available_categories.length || 0} categories available`}>
            <div className="button-row">
              <button disabled={!canExecuteCommand("workers.scan")} onClick={() => void runCommand("workers.scan")}>Rescan Workers</button>
              <button disabled={!canExecuteCommand("workers.services_reload")} onClick={() => void runCommand("workers.services_reload")}>Reload Services</button>
              <button disabled={!canExecuteCommand("workers.services_stop")} onClick={() => void runCommand("workers.services_stop")}>Stop Services</button>
              <button
                disabled={!canExecuteCommand("backend.reload_services")}
                onClick={() =>
                  void runCommand("backend.reload_services").then((result) =>
                    setToolResult(JSON.stringify(result ?? {}, null, 2)),
                  )
                }
              >
                Reload Backend Services
              </button>
              <button
                disabled={!canExecuteCommand("backend.watch_reload_start")}
                onClick={() =>
                  void runCommand("backend.watch_reload_start", {
                    kwargs: { interval_seconds: 1.0 },
                  }).then((result) => setToolResult(JSON.stringify(result ?? {}, null, 2)))
                }
              >
                Start Auto Reload
              </button>
              <button
                disabled={!canExecuteCommand("backend.watch_reload_stop")}
                onClick={() =>
                  void runCommand("backend.watch_reload_stop").then((result) =>
                    setToolResult(JSON.stringify(result ?? {}, null, 2)),
                  )
                }
              >
                Stop Auto Reload
              </button>
            </div>
            <div className="hint-row">
              runtime `{workerRuntimeMode}` | active services {workerServiceCount} | reloadable backend services{" "}
              {reloadableRuntimeServices.length} | auto reload {runtimeServiceWatch?.enabled ? "on" : "off"}
            </div>
            <div className="stack">
              {runtimeServiceWatch ? (
                <div className="tag-section">
                  <strong>Backend Auto Reload</strong>
                  <div className="path-pill">
                    interval {runtimeServiceWatch.interval_seconds ?? "-"}s | last error{" "}
                    {runtimeServiceWatch.last_error || "none"}
                  </div>
                </div>
              ) : null}
              {reloadableRuntimeServices.length ? (
                <div className="tag-section">
                  <strong>Reloadable Backend Services</strong>
                  <div className="tag-cloud">
                    {reloadableRuntimeServices.map(([name, moduleName]) => (
                      <span key={name} className="tag-chip">
                        {name}: {moduleName}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
              {workers?.service_manager?.services?.length ? (
                <div className="tag-section">
                  <strong>Active Worker Services</strong>
                  <div className="tag-cloud">
                    {workers.service_manager.services.map((service) => (
                      <span key={`${service.category}-${service.worker_name}`} className="tag-chip">
                        {service.category}/{service.worker_name}#{service.pid ?? "?"}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
              {workerCards.length ? (
                workerCards.map(([category, items]) => (
                  <div key={category} className="tag-section">
                    <strong>{category}</strong>
                    <div className="tag-cloud">
                      {items.map((worker) => (
                        <span key={`${category}-${worker.name}`} className="tag-chip">
                          {worker.display_name}
                        </span>
                      ))}
                    </div>
                  </div>
                ))
              ) : (
                <div className="empty-state">No workers discovered yet.</div>
              )}
            </div>
          </Panel>

          <Panel
            title="Runtime Tools"
            subtitle="Direct command execution for agent-style editing flows"
          >
            <div className="stack compact">
              <div className="tool-form">
                <strong>Custom Tag</strong>
                <div className="tool-row">
                  <input
                    value={customTagDraft}
                    onChange={(event) => setCustomTagDraft(event.target.value)}
                    placeholder="dramatic lighting"
                  />
                  <button
                    onClick={() =>
                      void runCommand("tags.add_custom", {
                        kwargs: { tag: customTagDraft },
                      }).then((result) => {
                        setToolResult(JSON.stringify(result ?? {}, null, 2));
                        setCustomTagDraft("");
                      })
                    }
                    disabled={!currentImagePath || !customTagDraft.trim() || !canExecuteCommand("tags.add_custom")}
                  >
                    Add
                  </button>
                </div>
              </div>

              <div className="tool-form">
                <strong>Find / Replace</strong>
                <div className="tool-grid">
                  <input
                    value={findTextDraft}
                    onChange={(event) => setFindTextDraft(event.target.value)}
                    placeholder="find text"
                  />
                  <input
                    value={replaceTextDraft}
                    onChange={(event) => setReplaceTextDraft(event.target.value)}
                    placeholder="replace text"
                  />
                </div>
                <div className="tool-row checkboxes">
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={findScopeAll}
                      onChange={(event) => setFindScopeAll(event.target.checked)}
                    />
                    <span>All images</span>
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={findCaseSensitive}
                      onChange={(event) => setFindCaseSensitive(event.target.checked)}
                    />
                    <span>Case sensitive</span>
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={findRegex}
                      onChange={(event) => setFindRegex(event.target.checked)}
                    />
                    <span>Regex</span>
                  </label>
                  <button
                    onClick={() =>
                      void runCommand("editor.find_replace", {
                        kwargs: {
                          find_text: findTextDraft,
                          replace_text: replaceTextDraft,
                          scope_all: findScopeAll,
                          case_sensitive: findCaseSensitive,
                          regex: findRegex,
                        },
                      }).then((result) => setToolResult(JSON.stringify(result ?? {}, null, 2)))
                    }
                    disabled={!findTextDraft.trim() || !canExecuteCommand("editor.find_replace")}
                  >
                    Apply Replace
                  </button>
                </div>
              </div>

              <div className="tool-form">
                <strong>Stroke Eraser</strong>
                <div className="tool-row">
                  <input
                    type="file"
                    accept="image/png,image/*"
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (!file) {
                        setStrokeMaskName("");
                        setStrokeMaskDataUrl("");
                        return;
                      }
                      void readFileAsDataUrl(file)
                        .then((dataUrl) => {
                          setStrokeMaskName(file.name);
                          setStrokeMaskDataUrl(dataUrl);
                          setError(null);
                        })
                        .catch((nextError) => {
                          setStrokeMaskName("");
                          setStrokeMaskDataUrl("");
                          setError(describeError(nextError));
                        });
                    }}
                  />
                  <button
                    onClick={() =>
                      void runCommand("action.run_stroke_eraser_current", {
                        kwargs: { mask_png_base64: strokeMaskDataUrl },
                      }).then((result) => setToolResult(JSON.stringify(result ?? {}, null, 2)))
                    }
                    disabled={!currentImagePath || !strokeMaskDataUrl || !canExecuteCommand("action.run_stroke_eraser_current")}
                  >
                    Apply Mask
                  </button>
                </div>
                <div className="path-pill">
                  {strokeMaskName || "Upload a white=erase, black=keep mask image"}
                </div>
              </div>

              <div className="tool-form">
                <strong>Danger Zone</strong>
                <div className="tool-row">
                  <button
                    className="danger"
                    onClick={() => {
                      if (!currentImagePath || !window.confirm("Move current image to no_used?")) {
                        return;
                      }
                      void runCommand("image.delete_current", {
                        kwargs: { confirm: true },
                      }).then((result) => setToolResult(JSON.stringify(result ?? {}, null, 2)));
                    }}
                    disabled={!currentImagePath || !canExecuteCommand("image.delete_current")}
                  >
                    Move Current To no_used
                  </button>
                </div>
              </div>

              <div className="event-card">
                <div className="event-head">
                  <strong>Last Tool Result</strong>
                  <span>{toolResult ? "latest" : "idle"}</span>
                </div>
                <pre>{toolResult || "{}"}</pre>
              </div>
            </div>
          </Panel>

          <Panel
            title="UI Spec Override"
            subtitle={capabilities?.ui_spec_override_active ? "override active" : "using base spec"}
          >
            <div className="button-row">
              <button
                disabled={!canExecuteCommand("app.patch_ui_node")}
                onClick={() =>
                  void runCommand("app.patch_ui_node", {
                    kwargs: { node_id: "text_editor_panel", patch: { hidden: true } },
                  })
                }
              >
                Hide Text Panel
              </button>
              <button
                disabled={!canExecuteCommand("app.patch_ui_node")}
                onClick={() =>
                  void runCommand("app.patch_ui_node", {
                    kwargs: { node_id: "text_editor_panel", patch: { hidden: false } },
                  })
                }
              >
                Show Text Panel
              </button>
              <button
                disabled={!canExecuteCommand("app.patch_ui_node")}
                onClick={() =>
                  void runCommand("app.patch_ui_node", {
                    kwargs: { node_id: "nl_tab", patch: { title: "Prompt Studio" } },
                  })
                }
              >
                Rename NL Tab
              </button>
              <button
                className="ghost"
                disabled={!canExecuteCommand("app.reset_ui_spec")}
                onClick={() =>
                  void runCommand("app.reset_ui_spec").then(() => setUiSpecDirty(false))
                }
              >
                Reset
              </button>
            </div>
            <EditorCard
              title="Override JSON"
              value={uiSpecDraft}
              dirty={uiSpecDirty}
              rows={12}
              disabled={!canExecuteCommand("app.replace_ui_spec_override")}
              onChange={(value) => {
                setUiSpecDraft(value);
                setUiSpecDirty(true);
              }}
              onSave={() => {
                let parsed: Record<string, unknown> = {};
                try {
                  parsed = uiSpecDraft.trim() ? (JSON.parse(uiSpecDraft) as Record<string, unknown>) : {};
                } catch (nextError) {
                  setError(describeError(nextError));
                  return;
                }
                void runCommand("app.replace_ui_spec_override", {
                  kwargs: { override: parsed as unknown as JsonObject },
                }).then(() => setUiSpecDirty(false));
              }}
            />
          </Panel>

          <Panel
            title="Capabilities"
            subtitle={`${agentSurfaceMode} | commands ${capabilities?.commands.length || 0}, destructive ${destructiveCapabilityCount}, spec ${capabilities?.ui_spec_version || "?"}`}
          >
            <div className="tag-cloud">
              {(capabilities?.commands || []).slice(0, 18).map((command) => (
                <span key={command} className="tag-chip">
                  {command}
                </span>
              ))}
            </div>
            <div className="tag-cloud">
              {(capabilities?.control_ids || []).map((controlId) => (
                <span key={controlId} className="tag-chip">
                  {controlId}
                </span>
              ))}
            </div>
            <div className="stack compact">
              {Object.entries(capabilities?.command_examples || {}).map(([commandName, example]) => (
                <CommandMetadataCard
                  key={commandName}
                  commandName={commandName}
                  metadata={capabilityCommandMetadata[commandName]}
                  example={example}
                />
              ))}
            </div>
          </Panel>

          <Panel
            title="Agent Surface"
            subtitle={agentManifest?.recommended_event_profile ? `${agentManifest.agent_mode} | profile ${agentManifest.recommended_event_profile}` : "manifest unavailable"}
          >
            <div className="stack compact">
              <div className="tab-strip">
                {(agentManifest?.available_modes || capabilities?.agent_modes || ["development"]).map((mode) => (
                  <button
                    key={mode}
                    className={mode === agentSurfaceMode ? "active" : ""}
                    onClick={() => setAgentSurfaceMode(mode)}
                  >
                    {mode}
                  </button>
                ))}
              </div>
              <div className="event-card">
                <div className="event-head">
                  <strong>Mode</strong>
                  <span>{agentManifest?.agent_mode || "unknown"}</span>
                </div>
                <pre>{agentManifest?.skills_note || ""}</pre>
              </div>
              <div className="tag-cloud">
                <span className="tag-chip">allowed {agentManifest?.allowed_commands.length || 0}</span>
                <span className="tag-chip">restricted {agentManifest?.restricted_commands.length || 0}</span>
                <span className="tag-chip">ui mutate {agentManifest?.ui_mutation_allowed ? "yes" : "no"}</span>
                <span className="tag-chip">settings mutate {agentManifest?.settings_mutation_allowed ? "yes" : "no"}</span>
                <span className="tag-chip">destructive ops {agentManifest?.destructive_file_ops_allowed ? "yes" : "no"}</span>
              </div>
              <div className="tag-cloud">
                {(agentManifest?.recommended_commands || []).slice(0, 16).map((command) => (
                  <span key={command} className="tag-chip">
                    {command}
                  </span>
                ))}
              </div>
              <div className="stack compact">
                {recommendedAgentCommands.map((commandName) => (
                  <CommandMetadataCard
                    key={commandName}
                    commandName={commandName}
                    metadata={agentCommandMetadata[commandName]}
                    example={agentManifest?.command_examples?.[commandName]}
                  />
                ))}
              </div>
              {agentManifest?.restricted_commands?.length ? (
                <div className="event-card">
                  <div className="event-head">
                    <strong>Restricted Commands</strong>
                    <span>{agentManifest.restricted_commands.length}</span>
                  </div>
                  <pre>{JSON.stringify(agentManifest.restricted_commands, null, 2)}</pre>
                </div>
              ) : null}
              <div className="stack compact">
                {restrictedAgentCommands.map((commandName) => (
                  <CommandMetadataCard
                    key={commandName}
                    commandName={commandName}
                    metadata={restrictedAgentCommandMetadata[commandName]}
                  />
                ))}
              </div>
              <div className="stack compact">
                {Object.entries(agentManifest?.routes || {}).map(([name, route]) => (
                  <div key={name} className="event-card">
                    <div className="event-head">
                      <strong>{name}</strong>
                      <span>{agentManifest?.endpoints?.[name] ? "endpoint" : "route"}</span>
                    </div>
                    <pre>{String(agentManifest?.endpoints?.[name] || route || "")}</pre>
                  </div>
                ))}
              </div>
            </div>
          </Panel>

          <Panel
            title="Settings Schema"
            subtitle={`${settingsSchema?.field_count || 0} fields across ${settingsSchema?.groups.length || 0} groups`}
          >
            <div className="tag-cloud">
              {(settingsSchema?.groups || []).map((group) => (
                <span key={group} className="tag-chip">
                  {group}
                </span>
              ))}
            </div>
            <div className="stack compact">
              {settingsPreview.map((field) => (
                <div key={field.key} className="event-card">
                  <div className="event-head">
                    <strong>{field.key}</strong>
                    <span>{field.type}</span>
                  </div>
                  <pre>{JSON.stringify(field.value, null, 2)}</pre>
                </div>
              ))}
            </div>
          </Panel>

          <Panel
            title="Settings Editor"
            subtitle={selectedSettingsGroup || "No group selected"}
          >
            <div className="tab-strip">
              {(settingsSchema?.groups || []).map((group) => (
                <button
                  key={group}
                  className={group === selectedSettingsGroup ? "active" : ""}
                  onClick={() => setSelectedSettingsGroup(group)}
                >
                  {group}
                </button>
              ))}
            </div>
            <div className="stack compact">
              {editableSettings.map((field) => (
                <SettingEditorRow
                  key={field.key}
                  field={field}
                  draftValue={settingDrafts[field.key]}
                  disabled={!canExecuteCommand("settings.update")}
                  onDraftChange={(value) =>
                    setSettingDrafts((current) => ({ ...current, [field.key]: value }))
                  }
                  onSave={() => {
                    let parsedValue: JsonValue;
                    try {
                      parsedValue = parseSettingDraft(field, settingDrafts[field.key]);
                    } catch (nextError) {
                      setError(describeError(nextError));
                      return;
                    }
                    void runCommand("settings.update", {
                      kwargs: { patch: { [field.key]: parsedValue } as JsonObject },
                    }).then(() =>
                      setSettingDrafts((current) => {
                        const nextDrafts = { ...current };
                        delete nextDrafts[field.key];
                        return nextDrafts;
                      }),
                    );
                  }}
                />
              ))}
            </div>
          </Panel>

          <Panel title="Runtime Events" subtitle="Latest 8 events">
            <div className="event-list">
              {visibleEvents.length ? (
                visibleEvents.map((event) => (
                  <div key={`${event.timestamp}-${event.name}`} className="event-card">
                    <div className="event-head">
                      <strong>{event.name}</strong>
                      <span>{new Date(event.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <pre>{JSON.stringify(event.payload, null, 2)}</pre>
                  </div>
                ))
              ) : (
                <div className="empty-state">No events yet.</div>
              )}
            </div>
          </Panel>

          <Panel title="Command Trace" subtitle="Latest started/finished/failed command summary">
            <div className="stack compact">
              <div className="event-card">
                <div className="event-head">
                  <strong>Last Started</strong>
                  <span>{runtimeState?.commands?.last_started?.timestamp ? new Date(runtimeState.commands.last_started.timestamp).toLocaleTimeString() : "idle"}</span>
                </div>
                <pre>{JSON.stringify(runtimeState?.commands?.last_started || {}, null, 2)}</pre>
              </div>
              <div className="event-card">
                <div className="event-head">
                  <strong>Last Finished</strong>
                  <span>{runtimeState?.commands?.last_finished?.timestamp ? new Date(runtimeState.commands.last_finished.timestamp).toLocaleTimeString() : "idle"}</span>
                </div>
                <pre>{JSON.stringify(runtimeState?.commands?.last_finished || {}, null, 2)}</pre>
              </div>
              <div className="event-card">
                <div className="event-head">
                  <strong>Last Failed</strong>
                  <span>{runtimeState?.commands?.last_failed?.timestamp ? new Date(runtimeState.commands.last_failed.timestamp).toLocaleTimeString() : "idle"}</span>
                </div>
                <pre>{JSON.stringify(runtimeState?.commands?.last_failed || {}, null, 2)}</pre>
              </div>
            </div>
          </Panel>

          <Panel title="Runtime State" subtitle="Snapshot">
            <pre className="state-dump">{JSON.stringify(runtimeState, null, 2)}</pre>
          </Panel>
          </section>
        </main>
      </details>

      {quickDialog ? (
        <div className="qt-dialog-backdrop" onClick={() => setQuickDialog("")}>
          <div className="qt-dialog" onClick={(event) => event.stopPropagation()}>
            <div className="qt-dialog-head">
              <strong>
                {quickDialog === "settings"
                  ? webText.settings
                  : quickDialog === "findReplace"
                    ? webText.findReplace
                    : quickDialog === "stroke"
                      ? webText.strokeEraser
                      : webText.addTag}
              </strong>
              <button className="ghost" onClick={() => setQuickDialog("")}>
                {webText.close}
              </button>
            </div>

            {quickDialog === "settings" ? (
              <div className="dialog-scroll">
                <div className="tab-strip">
                  {(settingsSchema?.groups || []).map((group) => (
                    <button
                      key={group}
                      className={group === selectedSettingsGroup ? "active" : ""}
                      onClick={() => setSelectedSettingsGroup(group)}
                    >
                      {group}
                    </button>
                  ))}
                </div>
                <div className="stack compact">
                  {editableSettings.map((field) => (
                    <SettingEditorRow
                      key={field.key}
                      field={field}
                      draftValue={settingDrafts[field.key]}
                      disabled={!canExecuteCommand("settings.update")}
                      onDraftChange={(value) =>
                        setSettingDrafts((current) => ({ ...current, [field.key]: value }))
                      }
                      onSave={() => {
                        let parsedValue: JsonValue;
                        try {
                          parsedValue = parseSettingDraft(field, settingDrafts[field.key]);
                        } catch (nextError) {
                          setError(describeError(nextError));
                          return;
                        }
                        void runCommand("settings.update", {
                          kwargs: { patch: { [field.key]: parsedValue } as JsonObject },
                        }).then(() =>
                          setSettingDrafts((current) => {
                            const nextDrafts = { ...current };
                            delete nextDrafts[field.key];
                            return nextDrafts;
                          }),
                        );
                      }}
                    />
                  ))}
                </div>
              </div>
            ) : null}

            {quickDialog === "findReplace" ? (
              <div className="dialog-scroll">
                <div className="tool-form">
                  <div className="tool-grid">
                    <input
                      value={findTextDraft}
                      onChange={(event) => setFindTextDraft(event.target.value)}
                      placeholder="find text"
                    />
                    <input
                      value={replaceTextDraft}
                      onChange={(event) => setReplaceTextDraft(event.target.value)}
                      placeholder="replace text"
                    />
                  </div>
                  <div className="tool-row checkboxes">
                    <label className="checkbox-row">
                      <input
                        type="checkbox"
                        checked={findScopeAll}
                        onChange={(event) => setFindScopeAll(event.target.checked)}
                      />
                      <span>{webText.allImages}</span>
                    </label>
                    <label className="checkbox-row">
                      <input
                        type="checkbox"
                        checked={findCaseSensitive}
                        onChange={(event) => setFindCaseSensitive(event.target.checked)}
                      />
                      <span>{webText.caseSensitive}</span>
                    </label>
                    <label className="checkbox-row">
                      <input
                        type="checkbox"
                        checked={findRegex}
                        onChange={(event) => setFindRegex(event.target.checked)}
                      />
                      <span>{webText.regex}</span>
                    </label>
                  </div>
                  <div className="qt-dialog-actions">
                    <button className="ghost" onClick={() => setQuickDialog("")}>
                      {webText.close}
                    </button>
                    <button
                      disabled={!findTextDraft.trim() || !canExecuteCommand("editor.find_replace")}
                      onClick={() =>
                        void runCommand("editor.find_replace", {
                          kwargs: {
                            find_text: findTextDraft,
                            replace_text: replaceTextDraft,
                            scope_all: findScopeAll,
                            case_sensitive: findCaseSensitive,
                            regex: findRegex,
                          },
                        }).then((result) => {
                          setToolResult(JSON.stringify(result ?? {}, null, 2));
                          setQuickDialog("");
                        })
                      }
                    >
                      {webText.apply}
                    </button>
                  </div>
                </div>
              </div>
            ) : null}

            {quickDialog === "customTag" ? (
              <div className="dialog-scroll">
                <div className="tool-form">
                  <div className="tool-row">
                    <input
                      value={customTagDraft}
                      onChange={(event) => setCustomTagDraft(event.target.value)}
                      placeholder="dramatic lighting"
                    />
                  </div>
                  <div className="qt-dialog-actions">
                    <button className="ghost" onClick={() => setQuickDialog("")}>
                      {webText.close}
                    </button>
                    <button
                      disabled={!currentImagePath || !customTagDraft.trim() || !canExecuteCommand("tags.add_custom")}
                      onClick={() =>
                        void runCommand("tags.add_custom", {
                          kwargs: { tag: customTagDraft },
                        }).then((result) => {
                          setToolResult(JSON.stringify(result ?? {}, null, 2));
                          setCustomTagDraft("");
                          setQuickDialog("");
                        })
                      }
                    >
                      {webText.apply}
                    </button>
                  </div>
                </div>
              </div>
            ) : null}

            {quickDialog === "stroke" ? (
              <div className="dialog-scroll">
                <div className="tool-form">
                  <div className="tool-row">
                    <input
                      type="file"
                      accept="image/png,image/*"
                      onChange={(event) => {
                        const file = event.target.files?.[0];
                        if (!file) {
                          setStrokeMaskName("");
                          setStrokeMaskDataUrl("");
                          return;
                        }
                        void readFileAsDataUrl(file)
                          .then((dataUrl) => {
                            setStrokeMaskName(file.name);
                            setStrokeMaskDataUrl(dataUrl);
                            setError(null);
                          })
                          .catch((nextError) => {
                            setStrokeMaskName("");
                            setStrokeMaskDataUrl("");
                            setError(describeError(nextError));
                          });
                      }}
                    />
                  </div>
                  <div className="path-pill">
                    {strokeMaskName || webText.uploadMaskHint}
                  </div>
                  <div className="qt-dialog-actions">
                    <button className="ghost" onClick={() => setQuickDialog("")}>
                      {webText.close}
                    </button>
                    <button
                      disabled={!currentImagePath || !strokeMaskDataUrl || !canExecuteCommand("action.run_stroke_eraser_current")}
                      onClick={() =>
                        void runCommand("action.run_stroke_eraser_current", {
                          kwargs: { mask_png_base64: strokeMaskDataUrl },
                        }).then((result) => {
                          setToolResult(JSON.stringify(result ?? {}, null, 2));
                          setQuickDialog("");
                        })
                      }
                    >
                      {webText.apply}
                    </button>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function Panel(props: { title: string; subtitle?: string; children: ReactNode }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2>{props.title}</h2>
          {props.subtitle ? <p>{props.subtitle}</p> : null}
        </div>
      </div>
      <div className="panel-body">{props.children}</div>
    </section>
  );
}

function DesktopMenuBar(props: {
  containerRef: { current: HTMLElement | null };
  groups: DesktopMenuGroup[];
  activeMenu: DesktopMenuId;
  onActiveMenuChange: (menuId: DesktopMenuId) => void;
}) {
  return (
    <nav className="qt-menubar desktop" ref={props.containerRef}>
      {props.groups.map((group) => {
        const isOpen = props.activeMenu === group.id;
        return (
          <div
            key={group.id}
            className={`qt-menu ${isOpen ? "open" : ""}`}
            onMouseEnter={() => {
              if (props.activeMenu) {
                props.onActiveMenuChange(group.id);
              }
            }}
          >
            <button
              type="button"
              className={`qt-menu-trigger ${isOpen ? "active" : ""}`}
              onClick={() => props.onActiveMenuChange(isOpen ? "" : group.id)}
            >
              {group.label}
            </button>
            {isOpen ? (
              <div className="qt-menu-popup">
                {group.items.map((item) => (
                  <button
                    type="button"
                    key={`${group.id}-${item.label}`}
                    className={`qt-menu-item ${item.danger ? "danger" : ""}`}
                    disabled={item.disabled}
                    onClick={() => {
                      props.onActiveMenuChange("");
                      item.onSelect();
                    }}
                  >
                    <span>{item.label}</span>
                    {item.shortcut ? <span className="qt-menu-shortcut">{item.shortcut}</span> : null}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        );
      })}
    </nav>
  );
}

function MetricCard(props: { label: string; value: string; hint: string }) {
  return (
    <div className="metric-card">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
      <small>{props.hint}</small>
    </div>
  );
}

function CommandMetadataCard(props: {
  commandName: string;
  metadata?: CommandMetadata | null;
  example?: unknown;
}) {
  const labels = commandGuardrailLabels(props.metadata);
  const notes = props.metadata?.notes || [];
  return (
    <div className="event-card">
      <div className="event-head">
        <strong>{props.commandName}</strong>
        <span>{props.metadata?.risk_level || "low"} risk</span>
      </div>
      <div className="tag-cloud">
        <span className={`tag-chip ${commandRiskClass(props.metadata)}`}>
          risk {props.metadata?.risk_level || "low"}
        </span>
        {labels.map((label) => (
          <span key={`${props.commandName}-${label}`} className="tag-chip muted">
            {label}
          </span>
        ))}
      </div>
      {notes.length ? <pre>{notes.join("\n")}</pre> : null}
      {props.example !== undefined ? <pre>{JSON.stringify(props.example, null, 2)}</pre> : null}
    </div>
  );
}

function EditorCard(props: {
  title: string;
  value: string;
  dirty: boolean;
  rows?: number;
  disabled?: boolean;
  onChange: (value: string) => void;
  onSave: () => void;
}) {
  return (
    <div className="editor-card">
      <div className="editor-head">
        <strong>{props.title}</strong>
        <button className={props.dirty ? "active" : "ghost"} onClick={props.onSave} disabled={props.disabled}>
          {props.dirty ? "Save Changes" : "Synced"}
        </button>
      </div>
      <textarea rows={props.rows || 6} value={props.value} disabled={props.disabled} onChange={(event) => props.onChange(event.target.value)} />
    </div>
  );
}

function SettingEditorRow(props: {
  field: SettingField;
  draftValue: unknown;
  disabled?: boolean;
  onDraftChange: (value: unknown) => void;
  onSave: () => void;
}) {
  const effectiveValue =
    props.draftValue === undefined ? props.field.value : props.draftValue;
  const serialized = serializeSettingValue(effectiveValue);
  const dirty =
    props.draftValue !== undefined &&
    JSON.stringify(props.field.value) !== JSON.stringify(props.draftValue);
  const isLongText =
    props.field.type === "string" &&
    (String(props.field.key).includes("prompt") || serialized.length > 90);

  return (
    <div className="event-card">
      <div className="event-head">
        <div>
          <strong>{props.field.key}</strong>
          <div className="field-meta">
            {props.field.type}
            {props.field.options?.length ? ` | ${props.field.options.join(", ")}` : ""}
          </div>
        </div>
        <button className={dirty ? "active" : "ghost"} onClick={props.onSave} disabled={props.disabled}>
          {dirty ? "Apply" : "Sync"}
        </button>
      </div>

      {props.field.type === "boolean" ? (
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={Boolean(effectiveValue)}
            disabled={props.disabled}
            onChange={(event) => props.onDraftChange(event.target.checked)}
          />
          <span>{String(Boolean(effectiveValue))}</span>
        </label>
      ) : props.field.options?.length ? (
        <select
          className="select-input"
          value={serialized}
          disabled={props.disabled}
          onChange={(event) => props.onDraftChange(event.target.value)}
        >
          {props.field.options.map((option) => (
            <option key={String(option)} value={String(option)}>
              {String(option)}
            </option>
          ))}
        </select>
      ) : isLongText ? (
        <textarea
          rows={5}
          value={serialized}
          disabled={props.disabled}
          onChange={(event) => props.onDraftChange(event.target.value)}
        />
      ) : (
        <input
          value={serialized}
          disabled={props.disabled}
          onChange={(event) => props.onDraftChange(event.target.value)}
        />
      )}
    </div>
  );
}

function serializeSettingValue(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value, null, 2);
}

function parseSettingDraft(field: SettingField, value: unknown): JsonValue {
  const raw = value === undefined ? field.value : value;
  if (field.type === "boolean") {
    return Boolean(raw);
  }
  if (field.type === "integer") {
    const parsed = Number.parseInt(String(raw), 10);
    if (Number.isNaN(parsed)) {
      throw new Error(`Invalid integer for ${field.key}`);
    }
    return parsed;
  }
  if (field.type === "number") {
    const parsed = Number.parseFloat(String(raw));
    if (Number.isNaN(parsed)) {
      throw new Error(`Invalid number for ${field.key}`);
    }
    return parsed;
  }
  if (field.type === "array" || field.type === "object") {
    try {
      return JSON.parse(String(raw)) as JsonValue;
    } catch (error) {
      throw new Error(`Invalid JSON for ${field.key}`);
    }
  }
  return String(raw);
}

function resolveTabId(spec: LegacyUiSpec | null, index: number): string {
  const tabs = spec?.layout?.children?.[1] && "children" in spec.layout.children[1]
    ? (spec.layout.children[1] as { children?: UiSpecNode[] }).children?.[0]
    : undefined;
  const featureTabs = tabs && "tabs" in tabs && Array.isArray((tabs as UiSpecNode).tabs)
    ? ((tabs as UiSpecNode).tabs || []).filter((tab) => !tab.hidden)
    : [];
  return String(featureTabs[index]?.id || featureTabs[0]?.id || "tags_tab");
}
