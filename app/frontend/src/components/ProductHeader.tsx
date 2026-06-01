import { Activity, FlaskConical, ListChecks } from "lucide-react";

export type ProductView = "analyze" | "review" | "models";

const NAV_ITEMS = [
  {
    view: "analyze",
    label: "Analyze",
    icon: Activity,
  },
  {
    view: "review",
    label: "Review",
    icon: ListChecks,
  },
  {
    view: "models",
    label: "Models",
    icon: FlaskConical,
  },
] as const;

type ProductHeaderProps = {
  activeView: ProductView;
  onViewChange: (view: ProductView) => void;
};

export function ProductHeader({ activeView, onViewChange }: ProductHeaderProps) {
  return (
    <header className="product-header">
      <a
        className="product-brand"
        href="#"
        aria-label="FoodLens home"
        onClick={(event) => {
          event.preventDefault();
          onViewChange("analyze");
        }}
      >
        FoodLens
      </a>
      <nav className="product-nav" aria-label="Product navigation">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = item.view === activeView;

          return (
            <button
              key={item.label}
              type="button"
              className={isActive ? "is-active" : ""}
              aria-current={isActive ? "page" : undefined}
              onClick={() => onViewChange(item.view)}
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
