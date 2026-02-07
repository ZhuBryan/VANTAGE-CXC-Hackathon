import reflex as rx
import random

class State(rx.State):
    current_company: str = "Global Logistics"
    hypocrisy_score: str = "40-55%"
    logs: list[str] = ["SYSTEM INITIALIZED...", "SATELLITE SYNCED: SECTOR 7-G"]

    @rx.var
    def is_danger(self) -> bool:
        try:
            score_val = int(self.hypocrisy_score.split('-')[0].replace('%', ''))
            return score_val > 60
        except:
            return False

    def set_company(self, name: str, score: str):
        self.current_company = name
        self.hypocrisy_score = score
        self.logs.append(f"SWITCH FEED: {name}")

    def reset_state(self):
        self.current_company = "System Standby"
        self.hypocrisy_score = "0%"
        self.logs = ["SYSTEM REBOOTED", "SCORE RESET TO ZERO"]

    def run_deep_scan(self):
        val1 = random.randint(10, 85)
        val2 = val1 + random.randint(5, 10)
        self.hypocrisy_score = f"{val1}-{val2}%"
        self.logs.append(f"DEEP SCAN COMPLETED: {self.hypocrisy_score} DISCREPANCY DETECTED")

def glass_pane(child, danger_var, **kwargs):
    base_style = {
        "background": "rgba(3, 15, 13, 0.8)",
        "backdrop_filter": "blur(20px)",
        "border": rx.cond(
            danger_var, 
            "1px solid rgba(239, 68, 68, 0.7)", 
            "1px solid rgba(94, 234, 212, 0.3)"
        ),
        "border_radius": "12px",
        "padding": "2rem",
        "position": "relative",
        "box_shadow": rx.cond(
            danger_var, 
            "0 0 30px rgba(239, 68, 68, 0.25)", 
            "none"
        ),
        "transition": "all 0.4s ease-in-out",
    }
    custom_style = kwargs.pop("style", {})
    return rx.box(child, style={**base_style, **custom_style}, **kwargs)

def corner_brackets(danger_var):
    color_class = rx.cond(danger_var, "border-red-600", "border-emerald-400/70")
    return rx.fragment(
        rx.box(class_name=["absolute top-4 left-4 w-6 h-6 border-t-2 border-l-2", color_class]),
        rx.box(class_name=["absolute top-4 right-4 w-6 h-6 border-t-2 border-r-2", color_class]),
        rx.box(class_name=["absolute bottom-4 left-4 w-6 h-6 border-b-2 border-l-2", color_class]),
        rx.box(class_name=["absolute bottom-4 right-4 w-6 h-6 border-b-2 border-r-2", color_class]),
    )

def index() -> rx.Component:
    primary_red = "#ef4444"
    primary_emerald = "#5EEAD4"
    accent_color = rx.cond(State.is_danger, primary_red, primary_emerald)
    
    # --- ANIMATED BUTTON STYLE ---
    # Added scaling and a custom transition for a "snappy" feel
    button_base_style = {
        "width": "160px",
        "height": "50px",
        "font_family": "monospace",
        "font_size": "12px",
        "letter_spacing": "0.15em",
        "background_color": "rgba(255, 255, 255, 0.03)",
        "backdrop_filter": "blur(5px)",
        "transition": "transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s, border 0.3s, color 0.3s",
        "cursor": "pointer",
        "_hover": {
            "transform": "scale(1.05) translateY(-2px)",
            "background_color": "rgba(255, 255, 255, 0.08)",
        },
        "_active": {
            "transform": "scale(0.95)",
        }
    }

    return rx.box(
        # Injecting CSS Keyframes for the pulsing glow
        rx.el.style("""
            @keyframes pulse-red {
                0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
                100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
            }
            @keyframes pulse-emerald {
                0% { box-shadow: 0 0 0 0 rgba(94, 234, 212, 0.2); }
                70% { box-shadow: 0 0 0 10px rgba(94, 234, 212, 0); }
                100% { box-shadow: 0 0 0 0 rgba(94, 234, 212, 0); }
            }
            .danger-pulse { animation: pulse-red 2s infinite; }
            .normal-pulse { animation: pulse-emerald 3s infinite; }
        """),

        rx.box(
            style={
                "position": "fixed",
                "inset": "0",
                "z_index": "-10",
                "background_image": "url('/vantage_bg.png')",
                "background_size": "cover",
                "background_position": "center",
                "filter": rx.cond(State.is_danger, "sepia(1) saturate(5) hue-rotate(-50deg) contrast(1.2)", "none"),
                "transition": "filter 0.5s ease", 
            }
        ),
        rx.box(class_name="fixed inset-0 z-[-5] bg-black/75"),

        rx.flex(
            # --- SIDEBAR ---
            rx.box(
                glass_pane(
                    rx.vstack(
                        rx.heading("VANTAGE", style={"color": accent_color}, class_name="tracking-[0.4em] text-3xl font-light mb-8 transition-colors"),
                        rx.vstack(
                            # Sidebar buttons with hover slides
                            rx.foreach(
                                [
                                    ("ECOFLOW ENERGY", "15-20%"), 
                                    ("GLOBAL LOGISTICS", "40-55%"), 
                                    ("VERIDIAN TEXTILES", "30-45%")
                                ],
                                lambda x: rx.button(
                                    x[0], 
                                    on_click=lambda: State.set_company(x[0], x[1]),
                                    class_name="w-full h-12 justify-start px-6 bg-white/5 border-l-2 border-emerald-500/30 text-white/60 text-[9px] tracking-widest transition-all hover:border-emerald-400 hover:text-white hover:pl-8"
                                )
                            ),
                            spacing="3", width="100%"
                        ),
                        rx.spacer(),
                        rx.vstack(
                            rx.text(rx.cond(State.is_danger, "AI CORE: ALERT", "AI CORE: MONITORING"), style={"color": accent_color}, class_name="text-[10px] tracking-[0.3em] mb-2"),
                            rx.text(State.hypocrisy_score, style={"color": accent_color}, class_name="text-6xl font-black leading-none transition-all"),
                            rx.box(
                                rx.box(style={"width": State.hypocrisy_score}, class_name="h-full bg-red-600 shadow-[0_0_15px_red] transition-all duration-1000"),
                                class_name="w-full h-1 bg-black/60 mt-4 rounded-full overflow-hidden" 
                            ),
                            align_items="start", width="100%",
                        ),
                        height="100%"
                    ),
                    danger_var=State.is_danger,
                    height="100%",
                ),
                width="320px", height="calc(100vh - 3rem)", margin="1.5rem", flex_shrink="0"
            ),

            # --- MAIN HUD ---
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.vstack(
                            rx.heading(State.current_company, style={"color": accent_color}, size="8", class_name="font-bold transition-colors"),
                            rx.text(
                                rx.cond(State.is_danger, "CRITICAL DISCREPANCY DETECTED", "Satellite AI Discrepancy Analysis"), 
                                style={"color": rx.cond(State.is_danger, primary_red, "white/40")}, 
                                class_name="font-light text-sm transition-colors"
                            ),
                            align_items="start"
                        ),
                        rx.spacer(),
                        rx.hstack(
                            # RESET BUTTON - Red outline, slight scale on hover
                            rx.button(
                                "RESET", 
                                on_click=State.reset_state, 
                                style={
                                    **button_base_style,
                                    "border": "1px solid rgba(239, 68, 68, 0.4)",
                                    "color": primary_red,
                                },
                                class_name="hover:shadow-[0_0_15px_rgba(239,68,68,0.2)]"
                            ),
                            # DEEP SCAN BUTTON - Dynamic glow pulse
                            rx.button(
                                "DEEP SCAN", 
                                on_click=State.run_deep_scan, 
                                style={
                                    **button_base_style,
                                    "border": rx.cond(State.is_danger, f"1px solid {primary_red}", f"1px solid {primary_emerald}"),
                                    "color": accent_color,
                                },
                                # Apply the CSS pulse classes we defined above
                                class_name=rx.cond(State.is_danger, "danger-pulse", "normal-pulse hover:shadow-emerald-500/20")
                            ),
                            spacing="4"
                        ),
                        width="100%", margin_bottom="2rem"
                    ),

                    rx.hstack(
                        glass_pane(
                            rx.center(
                                corner_brackets(danger_var=State.is_danger),
                                rx.vstack(
                                    rx.text(rx.cond(State.is_danger, "SENSORS ALERTING", "LIVE SATELLITE FEED"), style={"color": accent_color}, class_name="font-bold tracking-[0.4em] text-[10px]"),
                                    rx.box(rx.icon(tag="scan", size=40, color=accent_color), 
                                           class_name="p-8 border border-white/10 rounded-full animate-pulse my-4"),
                                    rx.text("LAT 43.46 - LON -80.52", class_name="text-white/20 font-mono text-[9px]"),
                                ),
                                height="100%"
                            ),
                            danger_var=State.is_danger, width="60%", height="420px"
                        ),
                        glass_pane(
                            rx.box(
                                corner_brackets(danger_var=State.is_danger),
                                rx.scroll_area(
                                    rx.vstack(
                                        rx.foreach(State.logs, lambda log: rx.text(f"> {log}", style={"color": accent_color}, class_name="font-mono text-[10px] uppercase opacity-70")),
                                        align_items="start"
                                    ),
                                    height="360px"
                                )
                            ),
                            danger_var=State.is_danger, width="40%", height="420px"
                        ),
                        width="100%", spacing="6"
                    ),

                    rx.hstack(
                        glass_pane(
                            rx.vstack(
                                rx.hstack(rx.icon(tag="database", size=16, color=accent_color), rx.text("Evidence Trail", class_name="font-bold text-white text-xs")),
                                rx.box(rx.box(style={"width": State.hypocrisy_score}, 
                                              class_name=rx.cond(State.is_danger, "h-full bg-red-600 shadow-[0_0_10px_red]", "h-full bg-emerald-500")), 
                                       class_name="w-full h-8 bg-black/40 mt-4"),
                                align_items="start"
                            ),
                            danger_var=State.is_danger, width="50%"
                        ),
                        glass_pane(
                            rx.vstack(
                                rx.hstack(rx.icon(tag=rx.cond(State.is_danger, "shield-alert", "shield-check"), size=16, color=accent_color), 
                                          rx.text("Neural Verification", class_name="font-bold text-white text-xs")),
                                rx.box(class_name="w-full h-8 border border-white/5 mt-4 bg-white/5"),
                                align_items="start"
                            ),
                            danger_var=State.is_danger, width="50%"
                        ),
                        width="100%", spacing="6"
                    ),
                    rx.box(height="2rem"),
                    width="100%", spacing="8"
                ),
                overflow_y="auto", height="100vh", width="100%", padding="2rem", class_name="no-scrollbar" 
            ),
            width="100%", height="100vh"
        ),
        class_name="min-h-screen overflow-hidden",
    )

app = rx.App()
app.add_page(index)