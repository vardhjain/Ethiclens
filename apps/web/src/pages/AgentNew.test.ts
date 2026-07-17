import { describe, expect, it } from "vitest";
import { parseCsvHeaderLine } from "./AgentNew";

describe("parseCsvHeaderLine", () => {
  it("splits a plain unquoted header", () => {
    expect(parseCsvHeaderLine("race,flag,score")).toEqual([
      "race",
      "flag",
      "score",
    ]);
  });

  it("keeps a comma inside quotes as part of the field instead of splitting on it", () => {
    expect(parseCsvHeaderLine('"income, monthly",race')).toEqual([
      "income, monthly",
      "race",
    ]);
  });

  it("unescapes doubled quotes inside a quoted field", () => {
    expect(parseCsvHeaderLine('"say ""hi""",race')).toEqual([
      'say "hi"',
      "race",
    ]);
  });

  it("trims whitespace around unquoted fields", () => {
    expect(parseCsvHeaderLine(" race , flag ")).toEqual(["race", "flag"]);
  });

  it("returns a single empty field for an empty line", () => {
    expect(parseCsvHeaderLine("")).toEqual([""]);
  });
});
