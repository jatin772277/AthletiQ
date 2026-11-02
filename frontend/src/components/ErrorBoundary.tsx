import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("[AthletiQ UI Error]:", error, errorInfo);
  }

  public reset = () => {
    this.setState({ hasError: false, error: null });
    this.props.onReset?.();
  };

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div className="center-state" style={{ minHeight: "50vh" }}>
          <div className="card" style={{ maxWidth: 460, padding: 28, gap: 14 }}>
            <div className="eyebrow tb">View Rendering Notice</div>
            <h3 style={{ margin: "4px 0", fontSize: 22 }}>Something went wrong in this view</h3>
            <p className="muted" style={{ margin: 0, fontSize: 13.5 }}>
              {this.state.error?.message || "An unexpected error occurred while rendering analytics."}
            </p>
            <div style={{ display: "flex", gap: 10, marginTop: 10 }}>
              <button className="btn-volt" style={{ height: 40 }} onClick={this.reset}>
                Try again
              </button>
              <button className="btn-ghost" style={{ height: 40 }} onClick={() => { window.location.hash = "overview"; window.location.reload(); }}>
                Reload app
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
