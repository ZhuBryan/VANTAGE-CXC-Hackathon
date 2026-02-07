import reflex as rx
from rxconfig import config
from .components.stats import hypocrisy_card
from .components.audit_log import audit_log_terminal

class State(rx.State):
    """The app state."""
    logs: list[str] = [
        "System initialized.",
        "Satellite feed synced: Sector 4-Alpha",
    ]
    hypocrisy_score: int = 74

    def clear_logs(self):
        self.logs = ["System logs cleared. Ready for new scan."]
        self.hypocrisy_score = 0

    def run_scan(self):
        import random
        new_messages = [
            "Analyzing spectral signatures...",
            "Checking Snowflake for ESG discrepancies...",
            "Anomaly detected in reforestation data.",
            "Cortex AI: High hypocrisy probability found.",
            "Scanning next sector..."
        ]
        self.logs.append(random.choice(new_messages))
        self.hypocrisy_score = random.randint(40, 95)

def index() -> rx.Component:
    return rx.box(
        rx.flex(
            # --- SIDEBAR ---
            rx.vstack(
                rx.heading("VANTAGE", class_name="text-emerald-500 tracking-tighter text-3xl mb-10"),
                rx.button("Live Satellite", variant="ghost", class_name="w-full justify-start text-emerald-400 bg-emerald-500/10"),
                rx.button("Audit History", variant="ghost", class_name="w-full justify-start text-slate-400 hover:text-emerald-400"),
                rx.button("Risk Map", variant="ghost", class_name="w-full justify-start text-slate-400 hover:text-emerald-400"),
                rx.spacer(),
                rx.text("Status: Operational", class_name="text-[10px] text-emerald-600 font-mono"),
                class_name="w-64 h-screen p-6 bg-slate-950 border-r border-slate-900"
            ),
            
            # --- MAIN CONTENT ---
            rx.vstack( # This was missing in your last snippet
                # Header
                rx.hstack(
                    rx.vstack(
                        rx.heading("Environmental Integrity Dashboard", size="8", class_name="text-white"),
                        rx.text("Satellite AI & ESG Discrepancy Analysis", class_name="text-slate-500"),
                        align_items="start",
                    ),
                    rx.spacer(),
                    rx.button(
                        "Reset", 
                        on_click=State.clear_logs, 
                        variant="outline",
                        class_name="border-slate-800 text-slate-400 hover:bg-slate-900 px-6 py-2 rounded-xl transition-all mr-2" 
                    ),
                    rx.button(
                        "Initiate Scan",
                        on_click=State.run_scan, 
                        class_name="bg-emerald-600 hover:bg-emerald-500 text-white px-6 py-2 rounded-xl transition-all"
                    ),
                    class_name="w-full mb-8" 
                ),
                
                # SATELLITE IMAGE AREA
                rx.box(
                    rx.center(
                        rx.vstack(
                            rx.spinner(color="jade", thickness=2, size="3"),
                            rx.text("Waiting for Satellite Coordinates...", class_name="text-slate-600 mt-4 font-mono"),
                        )
                    ),
                    class_name="w-full h-[400px] bg-slate-900/30 border border-slate-800 rounded-3xl mb-6"
                ),
                
                # DATA SECTION
                rx.grid(
                    hypocrisy_card(State.hypocrisy_score, "Hypocrisy Score"),
                    hypocrisy_card(12, "Deforestation Risk"),
                    hypocrisy_card(42, "Confidence Level"),
                    columns="3",
                    spacing="4",
                    width="100%"
                ),
                
                # THE AUDIT TERMINAL
                rx.box(
                    audit_log_terminal(State.logs),
                    class_name="w-full mt-6"
                ),
                
                class_name="flex-1 p-10 bg-slate-950 overflow-y-auto"
            ),
            width="100%"
        ),
        class_name="min-h-screen bg-black"
    )

app = rx.App(
    theme=rx.theme(appearance="dark", has_background=True, accent_color="jade")
)
app.add_page(index)