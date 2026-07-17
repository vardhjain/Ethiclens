import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MantineProvider } from "@mantine/core";
import { describe, expect, it, vi } from "vitest";
import { QueryError } from "./QueryError";

function renderWithMantine(ui: React.ReactElement) {
  return render(<MantineProvider>{ui}</MantineProvider>);
}

describe("QueryError", () => {
  it("shows the error message and a default title", () => {
    renderWithMantine(
      <QueryError error={new Error("Network down")} onRetry={() => {}} />,
    );
    expect(screen.getByText("Couldn't load this data")).toBeInTheDocument();
    expect(screen.getByText("Network down")).toBeInTheDocument();
  });

  it("falls back to a generic message for a non-Error value", () => {
    renderWithMantine(<QueryError error="oops" onRetry={() => {}} />);
    expect(screen.getByText("Something went wrong.")).toBeInTheDocument();
  });

  it("uses a custom title when given one", () => {
    renderWithMantine(
      <QueryError
        error={new Error("x")}
        onRetry={() => {}}
        title="Couldn't load audits"
      />,
    );
    expect(screen.getByText("Couldn't load audits")).toBeInTheDocument();
  });

  it("calls onRetry when the Retry button is clicked", async () => {
    const onRetry = vi.fn();
    renderWithMantine(<QueryError error={new Error("x")} onRetry={onRetry} />);
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
