const POLICIES = [
  {
    label: "Auto-accept",
    description: "High confidence",
    tone: "auto",
  },
  {
    label: "Suggest",
    description: "Ranked choices",
    tone: "suggest",
  },
  {
    label: "Confirm",
    description: "User check",
    tone: "confirm",
  },
  {
    label: "Review",
    description: "Known risk",
    tone: "review",
  },
] as const;

export function DecisionPolicyStrip() {
  return (
    <section className="decision-policy" aria-label="Decision policy">
      {POLICIES.map((policy) => (
        <article
          key={policy.label}
          className={`decision-policy__tile decision-policy__tile--${policy.tone}`}
        >
          <span className="decision-policy__dot" aria-hidden="true" />
          <span>{policy.label}</span>
          <strong>{policy.description}</strong>
        </article>
      ))}
    </section>
  );
}
