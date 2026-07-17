import {
  Alert,
  Badge,
  Button,
  Card,
  FileInput,
  Group,
  Loader,
  MultiSelect,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  api,
  type DemoDataset,
  type SchemaInferenceProposal,
} from "../api/client";
import { QueryError } from "../components/QueryError";

// Mirrors the real pipeline stages (routers/agent.py's _run_and_persist), but the
// backend runs them as one request with no incremental progress events — this is
// perceived progress, not observed progress. Advances on a timer and holds at the
// last message rather than looping, so a genuinely slow run doesn't look stuck or
// cycle back to "Planning" while still working.
const RUN_STAGE_MESSAGES = [
  "Planning the audit…",
  "Computing fairness metrics…",
  "Measuring mitigation…",
  "Writing the narrative…",
];

function useRunStageMessage(active: boolean): string {
  const [index, setIndex] = useState(0);
  useEffect(() => {
    if (!active) {
      setIndex(0);
      return;
    }
    const id = setInterval(() => {
      setIndex((i) => Math.min(i + 1, RUN_STAGE_MESSAGES.length - 1));
    }, 2500);
    return () => clearInterval(id);
  }, [active]);
  return RUN_STAGE_MESSAGES[index];
}

function RecentAuditsCard() {
  const records = useQuery({
    queryKey: ["agent-records"],
    queryFn: () => api.listAgentRecords(),
  });

  if (records.isLoading) {
    return (
      <Card withBorder>
        <Loader size="sm" />
      </Card>
    );
  }
  if (records.isError) {
    return (
      <Card withBorder>
        <QueryError
          error={records.error}
          onRetry={() => records.refetch()}
          title="Couldn't load recent audits"
        />
      </Card>
    );
  }
  if (!records.data || records.data.length === 0) return null;

  return (
    <Card withBorder>
      <Stack gap="sm">
        <Text fw={600}>Recent audits</Text>
        <Table.ScrollContainer minWidth={500}>
          <Table highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Run</Table.Th>
                <Table.Th>Outcome column</Table.Th>
                <Table.Th>Composite</Table.Th>
                <Table.Th>Status</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {records.data.map((r) => (
                <Table.Tr key={r.record_id}>
                  <Table.Td>
                    <Link to={`/agent/records/${r.record_id}`}>
                      {new Date(r.created_at).toLocaleString()}
                    </Link>
                  </Table.Td>
                  <Table.Td>{r.outcome_column}</Table.Td>
                  <Table.Td>
                    <Group gap={4}>
                      <Text size="sm">
                        {r.composite_score?.toFixed(3) ?? "—"}
                      </Text>
                      {r.composite_band && (
                        <Badge size="xs" variant="light">
                          {r.composite_band}
                        </Badge>
                      )}
                    </Group>
                  </Table.Td>
                  <Table.Td>
                    {r.degraded ? (
                      <Badge size="xs" color="yellow">
                        Degraded
                      </Badge>
                    ) : (
                      <Badge size="xs" color={r.grounded ? "green" : "gray"}>
                        {r.grounded ? "Grounded" : "Numeric-only"}
                      </Badge>
                    )}
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
      </Stack>
    </Card>
  );
}

const FAVORABLE = "favorable";
const ADVERSE = "adverse";

// A quoted header containing a comma (e.g. `"income, monthly"`) must not be split on
// that comma — a naive `line.split(",")` mangles it into two garbage column names that
// don't match anything the backend (which parses the CSV properly via pandas) reports,
// leaving the proposal's Select/MultiSelect dropdowns silently empty for that column.
export function parseCsvHeaderLine(line: string): string[] {
  const fields: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (inQuotes) {
      if (char === '"') {
        if (line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        current += char;
      }
    } else if (char === '"') {
      inQuotes = true;
    } else if (char === ",") {
      fields.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  fields.push(current.trim());
  return fields;
}

async function parseCsvColumns(file: File): Promise<string[]> {
  const text = await file.text();
  const firstLine = text.split(/\r?\n/, 1)[0] ?? "";
  return parseCsvHeaderLine(firstLine).filter(Boolean);
}

function columnsFromProposal(p: SchemaInferenceProposal): string[] {
  const cols = new Set<string>([
    p.outcome_column,
    ...p.protected_attribute_columns,
    ...p.feature_columns,
  ]);
  if (p.true_label_column) cols.add(p.true_label_column);
  if (p.score_column) cols.add(p.score_column);
  return [...cols];
}

export function AgentNewPage() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [demoKey, setDemoKey] = useState<string | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [proposal, setProposal] = useState<SchemaInferenceProposal | null>(
    null,
  );
  const [analyzing, setAnalyzing] = useState(false);
  const [running, setRunning] = useState(false);
  const runStageMessage = useRunStageMessage(running);

  const demos = useQuery({
    queryKey: ["demo-datasets"],
    queryFn: () => api.demoDatasets(),
  });

  function resetSelection() {
    setFile(null);
    setDemoKey(null);
    setProposal(null);
    setColumns([]);
  }

  function pickDemo(dataset: DemoDataset) {
    resetSelection();
    setDemoKey(dataset.key);
    setProposal(dataset.proposal);
    setColumns(columnsFromProposal(dataset.proposal));
  }

  async function analyze() {
    if (!file) return;
    setAnalyzing(true);
    try {
      const [cols, result] = await Promise.all([
        parseCsvColumns(file),
        api.proposeSchema(file),
      ]);
      setColumns(cols);
      setProposal(result);
    } catch (e) {
      notifications.show({ color: "red", message: (e as Error).message });
    } finally {
      setAnalyzing(false);
    }
  }

  async function confirmAndRun() {
    if (!proposal || (!file && !demoKey)) return;
    setRunning(true);
    try {
      const result = demoKey
        ? await api.runDemoAudit(demoKey, proposal)
        : await api.runAudit(file as File, proposal);
      notifications.show({ color: "green", message: "Audit complete." });
      await qc.invalidateQueries({ queryKey: ["agent-records"] });
      nav(`/agent/records/${result.record_id}`);
    } catch (e) {
      notifications.show({ color: "red", message: (e as Error).message });
    } finally {
      setRunning(false);
    }
  }

  function updateProposal(patch: Partial<SchemaInferenceProposal>) {
    if (!proposal) return;
    setProposal({ ...proposal, ...patch });
  }

  return (
    <Stack maw={640}>
      <Title order={3}>Agent audit</Title>

      <RecentAuditsCard />

      <Card withBorder>
        <Stack>
          <Text fw={600}>Try a demo dataset</Text>
          <Text size="sm" c="dimmed">
            No upload needed — each one exercises a different part of the
            pipeline (a real measured mitigation, a projected-only fallback, and
            a no-ground-truth audit).
          </Text>
          <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="sm">
            {demos.data?.map((d) => (
              <Card
                key={d.key}
                component="button"
                type="button"
                withBorder
                padding="sm"
                onClick={() => pickDemo(d)}
                aria-pressed={demoKey === d.key}
                bg={demoKey === d.key ? "blue.0" : undefined}
                style={{ cursor: "pointer", textAlign: "left", width: "100%" }}
              >
                <Stack gap={4}>
                  <Text size="sm" fw={600}>
                    {d.title}
                  </Text>
                  <Text size="xs" c="dimmed">
                    {d.blurb}
                  </Text>
                </Stack>
              </Card>
            ))}
          </SimpleGrid>
        </Stack>
      </Card>

      <Card withBorder>
        <Stack>
          <Text fw={600}>Or upload your own predictions CSV</Text>
          <Text size="sm" c="dimmed">
            CSV should contain the model&apos;s predictions, a protected
            attribute (e.g. race, sex), and optionally true labels. 5MB / 50,000
            row limit.
          </Text>
          <FileInput
            label="Predictions CSV"
            placeholder="Choose file"
            accept=".csv"
            value={file}
            onChange={(f) => {
              resetSelection();
              setFile(f);
            }}
          />
          <Button loading={analyzing} disabled={!file} onClick={analyze}>
            Analyze columns
          </Button>
        </Stack>
      </Card>

      {proposal && (
        <Card withBorder>
          <Stack>
            <Group justify="space-between">
              <Title order={4}>Confirm the proposed schema</Title>
              {demoKey && <Badge variant="light">Demo dataset</Badge>}
            </Group>
            <Alert color="yellow" title="Review before running">
              This is the agent&apos;s proposal, not a final decision — check it
              carefully, especially the outcome direction. Getting it backwards
              inverts every fairness finding.
            </Alert>
            <Text size="sm" c="dimmed">
              Agent&apos;s reasoning: {proposal.reasoning}
            </Text>

            <MultiSelect
              label="Protected attribute columns"
              data={columns}
              value={proposal.protected_attribute_columns}
              onChange={(v) =>
                updateProposal({ protected_attribute_columns: v })
              }
            />
            <Select
              label="Outcome / prediction column"
              data={columns}
              value={proposal.outcome_column}
              onChange={(v) => v && updateProposal({ outcome_column: v })}
            />
            <TextInput
              label="What does a positive (1) outcome mean?"
              value={proposal.outcome_direction.positive_label_meaning}
              onChange={(e) =>
                updateProposal({
                  outcome_direction: {
                    ...proposal.outcome_direction,
                    positive_label_meaning: e.currentTarget.value,
                  },
                })
              }
            />
            <Select
              label="Is that positive outcome favorable or adverse to the subject?"
              data={[
                { value: FAVORABLE, label: "Favorable (e.g. loan approved)" },
                { value: ADVERSE, label: "Adverse (e.g. flagged high-risk)" },
              ]}
              value={
                proposal.outcome_direction.positive_is_favorable
                  ? FAVORABLE
                  : ADVERSE
              }
              onChange={(v) =>
                updateProposal({
                  outcome_direction: {
                    ...proposal.outcome_direction,
                    positive_is_favorable: v === FAVORABLE,
                  },
                })
              }
            />
            <Select
              label="True label column (if any)"
              data={columns}
              value={proposal.true_label_column ?? null}
              clearable
              placeholder="None — Equalized Odds will be skipped"
              onChange={(v) => updateProposal({ true_label_column: v || null })}
            />
            <Select
              label="Continuous prediction score column (if any)"
              data={columns}
              value={proposal.score_column ?? null}
              clearable
              placeholder="None — measured before/after mitigation will be unavailable"
              onChange={(v) => updateProposal({ score_column: v || null })}
            />
            <MultiSelect
              label="Feature columns"
              data={columns}
              value={proposal.feature_columns}
              onChange={(v) => updateProposal({ feature_columns: v })}
            />

            <Button loading={running} onClick={confirmAndRun}>
              Confirm &amp; run audit
            </Button>
            {running && (
              <Text size="xs" c="dimmed" ta="center">
                {runStageMessage}
              </Text>
            )}
          </Stack>
        </Card>
      )}
    </Stack>
  );
}
