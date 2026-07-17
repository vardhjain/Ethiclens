import {
  Alert,
  Badge,
  Button,
  Card,
  Group,
  Loader,
  Stack,
  Table,
  Text,
  Textarea,
  Title,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useParams } from "react-router-dom";
import { api, type AgentRecommendation, type MeasuredMitigation, type ScorecardMetric } from "../api/client";
import { QueryError } from "../components/QueryError";

interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  grounded?: boolean;
  degraded?: boolean;
  failed?: boolean;
}

const BAD_CLASSIFICATIONS = new Set(["FAIL", "Flagged"]);

function MetricCell({ metric }: { metric: ScorecardMetric | undefined }) {
  if (!metric || metric.value == null) {
    return (
      <Text size="sm" c="dimmed">
        {metric?.classification === "INSUFFICIENT_DATA" ? "Insufficient data" : "N/A"}
      </Text>
    );
  }
  const bad = metric.classification != null && BAD_CLASSIFICATIONS.has(metric.classification);
  return (
    <Group gap={6} wrap="nowrap">
      <Text size="sm">{metric.value.toFixed(3)}</Text>
      {metric.classification && (
        <Badge size="xs" color={bad ? "red" : "gray"} variant="light">
          {metric.classification}
        </Badge>
      )}
    </Group>
  );
}

function MeasuredMitigationChart({ measured }: { measured: MeasuredMitigation }) {
  const diData = [
    { name: "Before", value: measured.di_before },
    { name: "After", value: measured.di_after },
  ];
  const accData = [
    { name: "Before", value: measured.accuracy_before },
    { name: "After", value: measured.accuracy_after },
  ];
  return (
    <Stack>
      <Group>
        <Badge color="green">Measured on held-out data</Badge>
        <Badge variant="light">{measured.strategy}</Badge>
        {measured.crossed_threshold && <Badge color="teal">Crosses 0.80 DI threshold</Badge>}
      </Group>
      <Text size="sm" c="dimmed">
        {measured.group_label} — n={measured.n_test} held-out rows. Accuracy cost:{" "}
        {(measured.accuracy_cost * 100).toFixed(1)} pts.
      </Text>
      <Group grow align="flex-start">
        <div>
          <Text size="xs" fw={500} ta="center" mb={4}>
            Disparate Impact
          </Text>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={diData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 1.2]} />
              <Tooltip />
              <Bar dataKey="value">
                {diData.map((d, i) => (
                  <Cell key={i} fill={d.name === "Before" ? "#c0392b" : "#27ae60"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <Text size="xs" fw={500} ta="center" mb={4}>
            Accuracy
          </Text>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={accData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Bar dataKey="value" fill="#2980b9" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Group>
    </Stack>
  );
}

function ProjectedMitigationChart({ group, recs }: { group: string; recs: AgentRecommendation[] }) {
  const data = recs.map((r) => ({ name: r.strategy_name, improvement: r.estimated_di_improvement }));
  return (
    <Stack gap={4}>
      <Group>
        <Text size="sm" fw={500}>
          {group}
        </Text>
        <Badge color="gray" variant="light">
          Projected, not measured
        </Badge>
      </Group>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} layout="vertical" margin={{ left: 24 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis type="category" dataKey="name" width={180} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          <Bar dataKey="improvement" name="Projected DI improvement" fill="#8e44ad" />
        </BarChart>
      </ResponsiveContainer>
    </Stack>
  );
}

function MitigationCard({
  measured,
  groupEntries,
}: {
  measured: MeasuredMitigation | null;
  groupEntries: [string, AgentRecommendation[]][];
}) {
  if (!measured && groupEntries.length === 0) return null;
  return (
    <Card withBorder>
      <Text fw={600} mb="sm">
        Mitigation: before vs. after
      </Text>
      {measured ? (
        <MeasuredMitigationChart measured={measured} />
      ) : (
        <Stack gap="lg">
          <Text size="sm" c="dimmed">
            No continuous score column was provided, so this shows projected estimates only —
            not a measured before/after. Upload a score column to enable a measured chart.
          </Text>
          {groupEntries.map(([group, recs]) => (
            <ProjectedMitigationChart key={group} group={group} recs={recs} />
          ))}
        </Stack>
      )}
    </Card>
  );
}

export function AgentRecordPage() {
  const { id = "" } = useParams();
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const record = useQuery({
    queryKey: ["agent-record", id],
    queryFn: () => api.getAgentRecord(id),
  });

  const ask = useMutation({
    mutationFn: (q: string) => api.askAgentRecord(id, q),
    onSuccess: (res) => {
      setMessages((m) => [
        ...m,
        { role: "assistant", text: res.answer, grounded: res.grounded, degraded: res.degraded },
      ]);
    },
    onError: (e) => {
      notifications.show({ color: "red", message: (e as Error).message });
      // Keep the question visible and mark it failed, instead of deleting it — the
      // textarea is already cleared by send(), so removing it too would lose the
      // question entirely and force the user to retype it.
      setMessages((m) => m.map((msg, i) => (i === m.length - 1 ? { ...msg, failed: true } : msg)));
    },
  });

  function send(text: string) {
    const q = text.trim();
    if (!q) return;
    setMessages((m) => [...m, { role: "user", text: q }]);
    setQuestion("");
    ask.mutate(q);
  }

  function retry(text: string, index: number) {
    setMessages((m) => m.filter((_, i) => i !== index));
    send(text);
  }

  if (record.isLoading) return <Loader />;
  if (record.isError) {
    return (
      <QueryError
        error={record.error}
        onRetry={() => record.refetch()}
        title="Couldn't load this audit report"
      />
    );
  }
  if (!record.data) return <Loader />;
  const r = record.data;
  const sc = r.scorecard;
  const groupEntries = Object.entries(sc.recommendations);

  return (
    <Stack maw={860}>
      <Group justify="space-between">
        <Title order={3}>Agent audit report</Title>
        <Group>
          {sc.composite_band && <Badge variant="light">{sc.composite_band}</Badge>}
          <Badge color={r.grounded ? "green" : "gray"}>
            {r.grounded ? "Narrative grounded" : "Numeric-only"}
          </Badge>
          {r.degraded && <Badge color="yellow">Degraded</Badge>}
        </Group>
      </Group>

      <Card withBorder>
        <Text fw={600} mb="xs">
          Narrative
        </Text>
        <Text size="sm">{r.narrative}</Text>
        <Text fw={600} mt="md" mb="xs">
          Recommended mitigation
        </Text>
        <Text size="sm">{r.mitigation_summary}</Text>
      </Card>

      <Card withBorder>
        <Text fw={600} mb="xs">
          Metrics scope
        </Text>
        <Text size="sm" c="dimmed">
          {sc.metrics_reason}
        </Text>
        {sc.excluded_subgroups.length > 0 && (
          <Stack gap={4} mt="xs">
            {sc.excluded_subgroups.map((e, i) => (
              <Text key={i} size="xs" c="dimmed">
                Excluded {e.attribute}:{e.group_value} (n={e.n}) — {e.reason}
              </Text>
            ))}
          </Stack>
        )}
      </Card>

      <Card withBorder>
        <Table.ScrollContainer minWidth={650}>
          <Table>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Group</Table.Th>
                <Table.Th>n (priv/unpriv)</Table.Th>
                <Table.Th>Disparate Impact</Table.Th>
                <Table.Th>Statistical Parity Diff.</Table.Th>
                <Table.Th>Equalized Odds</Table.Th>
                <Table.Th>Flagged</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {sc.groups.map((g, i) => (
                <Table.Tr key={i}>
                  <Table.Td>{g.group_label}</Table.Td>
                  <Table.Td>
                    {g.n_privileged} / {g.n_unprivileged}
                  </Table.Td>
                  <Table.Td>
                    <MetricCell metric={g.metrics["disparate_impact"]} />
                  </Table.Td>
                  <Table.Td>
                    <MetricCell metric={g.metrics["spd"]} />
                  </Table.Td>
                  <Table.Td>
                    <MetricCell metric={g.metrics["equalized_odds"]} />
                  </Table.Td>
                  <Table.Td>
                    {g.flagged ? (
                      <Badge color="red">Flagged</Badge>
                    ) : (
                      <Badge color="green">OK</Badge>
                    )}
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
      </Card>

      <MitigationCard measured={sc.measured_mitigation} groupEntries={groupEntries} />

      <Card withBorder>
        <Text fw={600} mb="sm">
          Ask about this audit
        </Text>
        <Stack gap="xs" mb="sm">
          {messages.length === 0 && (
            <Text size="sm" c="dimmed">
              Ask e.g. &quot;Why was {sc.groups[0]?.group_label ?? "this group"} flagged?&quot;
            </Text>
          )}
          {messages.map((m, i) => (
            <Alert
              key={i}
              color={m.role === "user" ? (m.failed ? "red" : "blue") : m.degraded ? "yellow" : "gray"}
              variant="light"
              title={m.role === "user" ? "You" : "Agent"}
            >
              {m.text}
              {m.failed && (
                <Group gap="xs" mt={4}>
                  <Text size="xs" c="red">
                    Failed to send.
                  </Text>
                  <Button size="xs" variant="subtle" onClick={() => retry(m.text, i)}>
                    Retry
                  </Button>
                </Group>
              )}
            </Alert>
          ))}
          {ask.isPending && <Loader size="sm" />}
        </Stack>
        <Group align="flex-end">
          <Textarea
            flex={1}
            placeholder="Ask a question about this audit"
            value={question}
            onChange={(e) => setQuestion(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(question);
              }
            }}
            autosize
            minRows={1}
            maxRows={4}
          />
          <Button onClick={() => send(question)} loading={ask.isPending} disabled={!question.trim()}>
            Ask
          </Button>
        </Group>
      </Card>
    </Stack>
  );
}
