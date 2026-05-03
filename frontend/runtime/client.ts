import type {
  AgentManifest,
  BridgeStatus,
  CommandRequest,
  CommandResponse,
  CommandsSummary,
  LegacyUiSpec,
  RuntimeCapabilities,
  RuntimeErrorInfo,
  RuntimeEvent,
  RuntimeState,
  SettingsSchema,
  UiCaptureSaveItem,
  UiCaptureSaveResponse,
  WorkersSummary,
} from "./types";

export interface RuntimeEventStream {
  close: () => void;
}

export interface EventQueryOptions {
  profile?: string;
  includePrefixes?: string[];
  excludePrefixes?: string[];
}

export class CaptionRuntimeError extends Error {
  readonly info?: RuntimeErrorInfo;
  readonly statusCode?: number;

  constructor(message: string, info?: RuntimeErrorInfo, statusCode?: number) {
    super(message);
    this.name = "CaptionRuntimeError";
    this.info = info;
    this.statusCode = statusCode;
  }
}

function withEventQuery(baseUrl: string, options: EventQueryOptions = {}, limit?: number): string {
  const params = new URLSearchParams();
  if (options.profile) {
    params.set("profile", options.profile);
  }
  if (typeof limit === "number") {
    params.set("limit", String(limit));
  }
  for (const prefix of options.includePrefixes || []) {
    params.append("include_prefix", prefix);
  }
  for (const prefix of options.excludePrefixes || []) {
    params.append("exclude_prefix", prefix);
  }
  const query = params.toString();
  return query ? `${baseUrl}?${query}` : baseUrl;
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    let parsedError = "";
    let parsedInfo: RuntimeErrorInfo | undefined;
    try {
      const parsed = JSON.parse(text) as { error?: string; error_info?: RuntimeErrorInfo };
      parsedError = parsed.error || "";
      parsedInfo = parsed.error_info;
    } catch {
      parsedError = "";
    }
    if (parsedError) {
      throw new CaptionRuntimeError(parsedError, parsedInfo, response.status);
    }
    if (text) {
      throw new CaptionRuntimeError(text, undefined, response.status);
    }
    throw new CaptionRuntimeError(`HTTP ${response.status}`, undefined, response.status);
  }
  return (await response.json()) as T;
}

export class CaptionRuntimeClient {
  constructor(private readonly baseUrl: string) {}

  getCurrentPreviewUrl(cacheBuster: string | number = Date.now()): string {
    return `${this.baseUrl}/preview/current?t=${cacheBuster}`;
  }

  getEventStreamUrl(options: EventQueryOptions = {}): string {
    return withEventQuery(`${this.baseUrl}/events/stream`, options);
  }

  async health(): Promise<{ ok: boolean; bridge: BridgeStatus }> {
    const response = await fetch(`${this.baseUrl}/health`);
    return readJson(response);
  }

  async getBridgeStatus(): Promise<BridgeStatus> {
    const response = await fetch(`${this.baseUrl}/bridge`);
    return readJson(response);
  }

  async getCommands(mode = "development"): Promise<CommandsSummary> {
    const response = await fetch(`${this.baseUrl}/commands?mode=${encodeURIComponent(mode)}`);
    return readJson(response);
  }

  async getCapabilities(mode = "development"): Promise<RuntimeCapabilities> {
    const response = await fetch(`${this.baseUrl}/capabilities?mode=${encodeURIComponent(mode)}`);
    return readJson(response);
  }

  async getAgentManifest(mode = "development"): Promise<AgentManifest> {
    const response = await fetch(`${this.baseUrl}/agent/manifest?mode=${encodeURIComponent(mode)}`);
    return readJson(response);
  }

  async getState(): Promise<RuntimeState> {
    const response = await fetch(`${this.baseUrl}/state`);
    return readJson(response);
  }

  async getEvents(limit = 120, options: EventQueryOptions = {}): Promise<RuntimeEvent[]> {
    const response = await fetch(withEventQuery(`${this.baseUrl}/events`, options, limit));
    const payload = await readJson<{ events: RuntimeEvent[] }>(response);
    return payload.events;
  }

  async getUiSpec(): Promise<LegacyUiSpec> {
    const response = await fetch(`${this.baseUrl}/ui-spec`);
    return readJson(response);
  }

  async getUiSpecOverride(): Promise<Record<string, unknown>> {
    const response = await fetch(`${this.baseUrl}/ui-spec-override`);
    return readJson(response);
  }

  async getWorkers(): Promise<WorkersSummary> {
    const response = await fetch(`${this.baseUrl}/workers`);
    return readJson(response);
  }

  async getSettingsSchema(): Promise<SettingsSchema> {
    const response = await fetch(`${this.baseUrl}/settings-schema`);
    return readJson(response);
  }

  async saveUiCaptures(captures: UiCaptureSaveItem[]): Promise<UiCaptureSaveResponse> {
    const response = await fetch(`${this.baseUrl}/captures/ui`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ captures }),
    });
    return readJson(response);
  }

  async executeCommand<T = unknown>(
    commandName: string,
    request: CommandRequest = {},
    accessMode = "development",
  ): Promise<CommandResponse<T>> {
    const response = await fetch(`${this.baseUrl}/commands/${commandName}?mode=${encodeURIComponent(accessMode)}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });
    return readJson(response);
  }

  subscribeEvents(
    onEvent: (event: RuntimeEvent) => void,
    onStatus?: (status: "connecting" | "live" | "closed" | "error") => void,
    options: EventQueryOptions = {},
  ): RuntimeEventStream | null {
    if (typeof EventSource === "undefined") {
      return null;
    }

    const source = new EventSource(this.getEventStreamUrl(options));
    onStatus?.("connecting");

    source.onopen = () => {
      onStatus?.("live");
    };

    source.addEventListener("runtime", (message) => {
      try {
        const payload = JSON.parse((message as MessageEvent<string>).data) as RuntimeEvent;
        onEvent(payload);
      } catch (error) {
        onStatus?.("error");
      }
    });

    source.onerror = () => {
      onStatus?.("error");
    };

    return {
      close: () => {
        source.close();
        onStatus?.("closed");
      },
    };
  }
}
