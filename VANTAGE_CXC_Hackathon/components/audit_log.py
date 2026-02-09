import reflex as rx

def audit_log_terminal(logs):
    return rx.box(
        rx.vstack(
            # Terminal Header
            rx.hstack(
                rx.hstack(
                    rx.box(class_name="w-2.5 h-2.5 rounded-full bg-slate-800"),
                    rx.box(class_name="w-2.5 h-2.5 rounded-full bg-slate-800"),
                    rx.box(class_name="w-2.5 h-2.5 rounded-full bg-slate-800"),
                    spacing="2",
                ),
                rx.text("VANTAGE_CORE_ENGINE.sh", class_name="text-[10px] text-slate-500 font-mono ml-4 uppercase tracking-widest"),
                rx.spacer(),
                rx.badge("LIVE FEED", color_scheme="jade", variant="outline", class_name="text-[8px]"),
                class_name="w-full p-3 border-b border-slate-800/50 bg-slate-900/80"
            ),
            # Log Entries using rx.foreach
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(
                        logs,
                        lambda log: rx.hstack(
                            rx.text(">>", class_name="text-jade-500 font-mono text-xs"),
                            rx.text(log, class_name="text-slate-300 font-mono text-xs"),
                            spacing="2"
                        )
                    ),
                    class_name="p-4",
                    align_items="start",
                    spacing="1",
                ),
                height="180px",
                class_name="bg-black/20"
            ),
            spacing="0",
        ),
        class_name="w-full border border-slate-800 rounded-lg overflow-hidden"
    )