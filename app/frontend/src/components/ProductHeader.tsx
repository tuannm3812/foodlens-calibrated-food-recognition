import { Activity, FlaskConical, ListChecks } from "lucide-react";

const NAV_ITEMS = [
  {
    label: "Analyze",
    icon: Activity,
    active: true,
    disabled: false,
  },
  {
    label: "Review",
    icon: ListChecks,
    active: false,
    disabled: true,
  },
  {
    label: "Models",
    icon: FlaskConical,
    active: false,
    disabled: true,
  },
] as const;

export function ProductHeader() {
  return (
    <header className="product-header">
      <a className="product-brand" href="#" aria-label="FoodLens home">
        FoodLens
      </a>
      <nav className="product-nav" aria-label="Product navigation">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;

          return (
            <button
              key={item.label}
              type="button"
              className={item.active ? "is-active" : ""}
              aria-current={item.active ? "page" : undefined}
              disabled={item.disabled}
            >
              <Icon size={15} aria-hidden="true" />
              {item.label}
            </button>
          );
        })}
      </nav>
      <div className="product-header__meta" aria-label="Prototype status">
        <span>Prototype</span>
      </div>
    </header>
  );
}
