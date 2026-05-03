import { toPng } from "html-to-image";

import type { JsonObject, JsonValue, LegacyUiSpec } from "../runtime/types";

export type AgentCaptureScope =
  | "full_app"
  | "workspace"
  | "left_panel"
  | "right_panel"
  | "current_tab"
  | "text_editor"
  | "preview";

export interface AgentCaptureTabDescriptor {
  id: string;
  title: string;
}

export interface PreparedAgentCapture {
  name: string;
  scope: string;
  selector: string;
  png_data_url: string;
  metadata: JsonObject;
}

interface CapturePreset {
  width: number;
  mode: "normalized-desktop";
}

const TARGET_SELECTORS: Record<AgentCaptureScope, string> = {
  full_app: ".qt-web-app",
  workspace: ".qt-main-panel",
  left_panel: '[data-node-id="left_panel"]',
  right_panel: '[data-node-id="right_panel"]',
  current_tab: '.spec-tabs[data-node-id="feature_tabs"] .spec-tab',
  text_editor: '[data-node-id="text_editor_panel"]',
  preview: '[data-node-id="image_viewer"]',
};

const CAPTURE_PRESETS: Record<AgentCaptureScope, CapturePreset> = {
  full_app: { width: 1600, mode: "normalized-desktop" },
  workspace: { width: 1440, mode: "normalized-desktop" },
  left_panel: { width: 680, mode: "normalized-desktop" },
  right_panel: { width: 860, mode: "normalized-desktop" },
  current_tab: { width: 860, mode: "normalized-desktop" },
  text_editor: { width: 860, mode: "normalized-desktop" },
  preview: { width: 700, mode: "normalized-desktop" },
};

function roundValue(value: number): number {
  return Math.round(value * 10) / 10;
}

function normalizeText(value: string): string {
  return value.replace(/\s+/g, " ").trim().slice(0, 140);
}

function inferRole(element: HTMLElement): string {
  const explicitRole = element.getAttribute("role");
  if (explicitRole) {
    return explicitRole;
  }

  const tagName = element.tagName.toLowerCase();
  if (tagName === "button") {
    return "button";
  }
  if (tagName === "input") {
    const inputType = element.getAttribute("type") || "text";
    return inputType === "checkbox" ? "checkbox" : "textbox";
  }
  if (tagName === "textarea") {
    return "textbox";
  }
  if (tagName === "select") {
    return "combobox";
  }
  if (/^h[1-6]$/.test(tagName)) {
    return "heading";
  }
  if (tagName === "img") {
    return "img";
  }
  if (tagName === "nav") {
    return "navigation";
  }
  if (tagName === "main") {
    return "main";
  }
  if (tagName === "section") {
    return "region";
  }
  return "";
}

function inferName(element: HTMLElement): string {
  const ariaLabel = element.getAttribute("aria-label");
  if (ariaLabel) {
    return ariaLabel;
  }
  const title = element.getAttribute("title");
  if (title) {
    return title;
  }
  if (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement) {
    if (element.placeholder) {
      return element.placeholder;
    }
    if (element.value) {
      return normalizeText(element.value);
    }
  }
  const text = normalizeText(element.innerText || element.textContent || "");
  return text.slice(0, 90);
}

function isVisible(element: HTMLElement): boolean {
  const rect = element.getBoundingClientRect();
  const style = window.getComputedStyle(element);
  return (
    rect.width > 0 &&
    rect.height > 0 &&
    style.display !== "none" &&
    style.visibility !== "hidden" &&
    style.opacity !== "0"
  );
}

function buildSemanticSnapshot(
  element: HTMLElement,
  depth = 0,
  maxDepth = 4,
  maxChildren = 12,
): JsonObject | null {
  if (!isVisible(element)) {
    return null;
  }

  const rect = element.getBoundingClientRect();
  const children: JsonValue[] = [];
  if (depth < maxDepth) {
    for (const child of Array.from(element.children).slice(0, maxChildren)) {
      if (!(child instanceof HTMLElement)) {
        continue;
      }
      const childSnapshot = buildSemanticSnapshot(child, depth + 1, maxDepth, maxChildren);
      if (childSnapshot) {
        children.push(childSnapshot);
      }
    }
  }

  const role = inferRole(element);
  const name = inferName(element);
  const node: JsonObject = {
    tag: element.tagName.toLowerCase(),
    bounds: {
      x: roundValue(rect.x + window.scrollX),
      y: roundValue(rect.y + window.scrollY),
      width: roundValue(rect.width),
      height: roundValue(rect.height),
    } as unknown as JsonValue,
  };

  if (role) {
    node.role = role;
  }
  if (name) {
    node.name = name;
  }
  if (children.length) {
    node.children = children;
  }

  return node;
}

export function listCaptureTabs(spec: LegacyUiSpec | null): AgentCaptureTabDescriptor[] {
  if (!spec) {
    return [];
  }

  const queue = [spec.layout];
  while (queue.length) {
    const node = queue.shift();
    if (!node) {
      continue;
    }
    if (node.type === "tabs" && Array.isArray(node.tabs)) {
      return node.tabs
        .filter((tab) => !tab.hidden)
        .map((tab) => ({
          id: String(tab.id || ""),
          title: String(tab.title || tab.id || "tab"),
        }))
        .filter((tab) => tab.id);
    }
    for (const child of node.children || []) {
      queue.push(child);
    }
    for (const child of node.sections || []) {
      queue.push(child);
    }
    for (const child of node.tabs || []) {
      queue.push(child);
    }
  }
  return [];
}

export function getCaptureSelector(scope: AgentCaptureScope): string {
  return TARGET_SELECTORS[scope];
}

export function findCaptureElement(scope: AgentCaptureScope): HTMLElement | null {
  return document.querySelector<HTMLElement>(getCaptureSelector(scope));
}

function copyFormState(sourceRoot: HTMLElement, cloneRoot: HTMLElement): void {
  const sourceControls = Array.from(
    sourceRoot.querySelectorAll<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(
      "input, textarea, select",
    ),
  );
  const cloneControls = Array.from(
    cloneRoot.querySelectorAll<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(
      "input, textarea, select",
    ),
  );

  for (let index = 0; index < Math.min(sourceControls.length, cloneControls.length); index += 1) {
    const source = sourceControls[index];
    const clone = cloneControls[index];
    if (source instanceof HTMLInputElement && clone instanceof HTMLInputElement) {
      clone.value = source.value;
      clone.checked = source.checked;
    } else if (source instanceof HTMLTextAreaElement && clone instanceof HTMLTextAreaElement) {
      clone.value = source.value;
    } else if (source instanceof HTMLSelectElement && clone instanceof HTMLSelectElement) {
      clone.value = source.value;
    }
  }
}

async function createNormalizedCaptureNode(
  sourceElement: HTMLElement,
  scope: AgentCaptureScope,
): Promise<{ sandbox: HTMLDivElement; captureNode: HTMLElement; liveRect: DOMRect }> {
  const preset = CAPTURE_PRESETS[scope];
  const sandbox = document.createElement("div");
  sandbox.className = "agent-capture-sandbox";

  const clone = sourceElement.cloneNode(true) as HTMLElement;
  copyFormState(sourceElement, clone);
  clone.classList.add("agent-capture-root", "agent-capture-normalized");
  clone.setAttribute("data-capture-scope", scope);
  clone.setAttribute("data-capture-mode", preset.mode);
  clone.style.width = `${preset.width}px`;
  clone.style.minWidth = `${preset.width}px`;
  clone.style.maxWidth = `${preset.width}px`;

  sandbox.appendChild(clone);
  document.body.appendChild(sandbox);

  await new Promise<void>((resolve) => {
    window.requestAnimationFrame(() => resolve());
  });

  return {
    sandbox,
    captureNode: clone,
    liveRect: sourceElement.getBoundingClientRect(),
  };
}

export async function prepareAgentCapture(
  scope: AgentCaptureScope,
  name: string,
  metadata: JsonObject,
): Promise<PreparedAgentCapture> {
  const element = findCaptureElement(scope);
  if (!element) {
    throw new Error(`Capture target '${scope}' is not available.`);
  }

  const preset = CAPTURE_PRESETS[scope];
  const { sandbox, captureNode, liveRect } = await createNormalizedCaptureNode(element, scope);
  try {
    const captureRect = captureNode.getBoundingClientRect();
    const pngDataUrl = await toPng(captureNode, {
      cacheBust: true,
      pixelRatio: 1,
      backgroundColor: "#eeeeee",
      width: Math.round(captureRect.width),
      height: Math.round(captureRect.height),
    });

    return {
      name,
      scope,
      selector: getCaptureSelector(scope),
      png_data_url: pngDataUrl,
      metadata: {
        ...metadata,
        capture_mode: preset.mode,
        capture_target: {
          scope,
          selector: getCaptureSelector(scope),
          bounds: {
            x: roundValue(liveRect.x + window.scrollX),
            y: roundValue(liveRect.y + window.scrollY),
            width: roundValue(liveRect.width),
            height: roundValue(liveRect.height),
          },
          baseline_bounds: {
            width: roundValue(captureRect.width),
            height: roundValue(captureRect.height),
          },
          baseline_width_px: preset.width,
        } as unknown as JsonValue,
        capture_environment: {
          device_pixel_ratio: window.devicePixelRatio,
          visual_viewport_scale: window.visualViewport?.scale ?? 1,
          inner_width: window.innerWidth,
          inner_height: window.innerHeight,
        } as unknown as JsonValue,
        semantic_snapshot: buildSemanticSnapshot(element),
      },
    };
  } finally {
    sandbox.remove();
  }
}
