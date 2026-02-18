import reflex as rx
import httpx
import asyncio
import random
import json


class State(rx.State):
    current_company: str = "System Standby"
    hypocrisy_score: str = "0%"
    logs: list[str] = ["SYSTEM INITIALIZED...", "SATELLITE SYNCED: SECTOR 7-G"]

    # Map State
    lat: float = 43.46
    lon: float = -80.52
    zoom: int = 14
    map_url: str = "https://maps.google.com/maps?q=43.46,-80.52&z=14&output=embed&t=k"

    # View Mode: 'MAP' or 'HEATMAP'
    view_mode: str = "MAP"
    heatmap_url: str = ""

    # Deep Scan State
    is_deep_scanning: bool = False
    zoom_animation_active: bool = False
    heatmap_zoom_level: float = 1.0

    # Document Analysis State (Gemini)
    is_analyzing: bool = False
    show_report: bool = False
    gemini_summary: str = ""
    gemini_claims: list[dict] = []
    gemini_red_flags: list[str] = []
    gemini_risk: str = ""
    gemini_recommendations: list[str] = []
    report_data: dict = {}

    @rx.var
    def is_danger(self) -> bool:
        try:
            score_val = int(self.hypocrisy_score.split("-")[0].replace("%", ""))
            return score_val > 60
        except Exception:
            return False

    @rx.var
    def active_map_class(self) -> str:
        return "zoom-active" if self.zoom_animation_active else ""

    def toggle_view(self):
        if self.view_mode == "MAP":
            if self.heatmap_url:
                self.view_mode = "HEATMAP"
        else:
            self.view_mode = "MAP"
            self.heatmap_zoom_level = 1.0
            self.zoom_animation_active = False  # Reset animation only when returning to map

    def set_company(self, name: str, score: str):
        self.current_company = name
        self.hypocrisy_score = score
        self.logs.append(f"SWITCH FEED: {name}")
        self.show_report = False
        self.report_data = {}
        self.view_mode = "MAP"
        self.heatmap_url = ""
        self.heatmap_zoom_level = 1.0
        self.gemini_summary = ""
        self.gemini_claims = []
        self.gemini_red_flags = []
        self.gemini_risk = ""
        self.gemini_recommendations = []

        if "Ford" in name:
            # 55°45'13"N 127°57'29"W -> 55.7536, -127.9581
            self.update_map(55.7536, -127.9581)
        elif "Drax" in name:
            # 55°45'41"N 127°58'35"W -> 55.7614, -127.9764
            # Updated to: 55.735425770292935, -128.0205007766051
            self.update_map(55.735425770292935, -128.0205007766051)
        elif "Veridian" in name:
            # 31°40'59"N 99°34'01"W -> 31.6831, -99.5669
            self.update_map(31.6831, -99.5669)
        else:
            self.update_map(43.46, -80.52)

    def update_map(self, lat, lon):
        self.lat = lat
        self.lon = lon
        # Force a refresh of the map URL
        self.map_url = f"https://maps.google.com/maps?q={lat},{lon}&z={self.zoom}&output=embed&t=k"
        # Reset view mode to receive the new map
        self.view_mode = "MAP"
        self.heatmap_url = ""
        self.is_deep_scanning = False
        self.zoom_animation_active = False

    def reset_state(self):
        self.current_company = "System Standby"
        self.hypocrisy_score = "0%"
        self.logs = ["SYSTEM REBOOTED", "SCORE RESET TO ZERO"]
        self.show_report = False
        self.is_analyzing = False
        self.is_deep_scanning = False
        self.zoom_animation_active = False
        self.view_mode = "MAP"
        self.heatmap_url = ""
        self.heatmap_zoom_level = 1.0
        self.gemini_summary = ""
        self.gemini_claims = []
        self.gemini_red_flags = []
        self.gemini_risk = ""
        self.gemini_recommendations = []
        self.update_map(43.46, -80.52)

    # --- DEEP SCAN WITH ZOOM ANIMATION ---
    async def run_deep_scan(self):
        # Prevent double-clicks
        if self.is_deep_scanning:
            return

        base_url = "http://localhost:8001"
        endpoint = "/demo/verified"

        if "EcoFlow" in self.current_company or "Drax" in self.current_company:
            endpoint = "/demo/industrial"
        elif "Veridian" in self.current_company:
            endpoint = "/demo/ghost"

        # Phase 1: Zoom animation
        self.is_deep_scanning = True
        self.zoom_animation_active = True
        self.logs.append("INITIATING DEEP SCAN SEQUENCE...")
        self.logs.append("SATELLITE ZOOM LOCK ENGAGED")
        yield

        # Artificial timing for the zoom animation (CSS is 2s)
        await asyncio.sleep(1.8) 

        # Phase 2: Call the API
        self.logs.append(f"CONNECTING TO VISION API... [{endpoint}]")
        yield

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(f"{base_url}{endpoint}")

            if response.status_code == 200:
                data = response.json()
                self.heatmap_url = data.get("heatmap_url", "")
                risk_level = data.get("risk_level", "LOW")
                
                # --- ZOOM FIX: KEEP ANIMATION ACTIVE ---
                # We do NOT turn off zoom_animation_active here.
                # Keeping it True ensures the map stays "zoomed in" while the heatmap overlay appears.
                # If we turn it False, the map snaps back to 1.0 scale before disappearing (The "Double Take").
                # self.zoom_animation_active = False  <-- REMOVED

                if risk_level == "CRITICAL":
                    final_score = random.randint(85, 99)
                    self.hypocrisy_score = f"{final_score}%"
                elif risk_level == "HIGH":
                    final_score = random.randint(65, 84)
                    self.hypocrisy_score = f"{final_score}%"
                else:
                    self.hypocrisy_score = f"{random.randint(0, 5)}%"

                self.logs.append(f"SCAN RESULT: {risk_level}")
                self.logs.append(f"MSG: {data.get('status_message')}")

                # Phase 3: Switch to heatmap immediately
                if self.heatmap_url:
                    self.heatmap_zoom_level = 1.0
                    self.view_mode = "HEATMAP"
                    # Animation resets only when user clicks "Return"
                    
            else:
                self.logs.append(f"API ERROR: {response.status_code}")
                self.zoom_animation_active = False

        except Exception as e:
            self.logs.append(f"CONNECTION ERROR: {str(e)}")
            val1 = random.randint(10, 85)
            self.hypocrisy_score = f"{val1}%"
            self.zoom_animation_active = False

        self.is_deep_scanning = False
        yield

    # --- HEATMAP ZOOM CONTROLS ---
    def zoom_heatmap_in(self):
        if self.heatmap_zoom_level < 6.0:  # Allow deeper zoom
            self.heatmap_zoom_level = min(self.heatmap_zoom_level + 0.5, 6.0)

    def zoom_heatmap_out(self):
        if self.heatmap_zoom_level > 1.0:
            self.heatmap_zoom_level = max(self.heatmap_zoom_level - 0.5, 1.0)

    # --- DOCUMENT UPLOAD & GEMINI ANALYSIS ---
    async def process_upload(self, files: list[rx.UploadFile]):
        """Upload document and analyze with Gemini API."""
        self.is_analyzing = True
        self.show_report = False
        self.logs.append("DOCUMENT RECEIVED — INITIATING NEURAL PARSE...")
        yield  # Show loading overlay

        # Determine satellite context for the prompt
        satellite_verdict = "No scan performed"
        risk_level = self.hypocrisy_score
        if self.view_mode == "HEATMAP":
            satellite_verdict = "Satellite anomaly detected" if self.is_danger else "Satellite imagery verified"

        try:
            # Read the uploaded file
            file = files[0]
            file_bytes = await file.read()
            filename = file.filename or "document.pdf"

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "http://localhost:8001/analyze-document",
                    files={"file": (filename, file_bytes)},
                    data={
                        "company_name": self.current_company,
                        "satellite_verdict": satellite_verdict,
                        "risk_level": risk_level,
                    },
                )

            if response.status_code == 200:
                data = response.json()
                self.gemini_summary = data.get("executive_summary", "")
                self.gemini_claims = data.get("claims_extracted", [])
                self.gemini_red_flags = data.get("red_flags", [])
                self.gemini_risk = data.get("overall_risk_score", "UNKNOWN")
                self.gemini_recommendations = data.get("recommendations", [])
                self.logs.append(f"ANALYSIS COMPLETE — RISK: {self.gemini_risk}")
            else:
                self.logs.append(f"ANALYSIS ERROR: HTTP {response.status_code}")
                self._fallback_report()

        except Exception as e:
            self.logs.append("GEMINI OFFLINE — USING CACHED INTELLIGENCE")
            self._fallback_report()

        self.is_analyzing = False
        self.show_report = True
        yield  # Show results

    def _fallback_report(self):
        """Hardcoded realistic reports for demo stability."""
        if "Ford" in self.current_company:
            self.gemini_summary = "Document analysis confirms Ford's 'Wild Fund' initiatives are supporting verified reforestation projects. Satellite data aligns with reported canopy growth."
            self.gemini_claims = [
                {"claim": "Investment in 500 acres of protected woodland", "category": "Conservation", "verdict": "VERIFIED", "reasoning": "Satellite confirms protected status and healthy vegetation index."},
                {"claim": "Carbon neutral assembly plant operations", "category": "Carbon Credits", "verdict": "VERIFIED", "reasoning": "Offsets matched to verified renewable projects."},
            ]
            self.gemini_red_flags = []
            self.gemini_risk = "LOW"
            self.gemini_recommendations = ["Continue quarterly satellite monitoring", "Highlight success in ESG report"]
        elif "Drax" in self.current_company:
            self.gemini_summary = "CRITICAL: Drax Report claims 'sustainable biomass' but satellite forensics reveal clear-cutting of designated ancient primary forest. Significant biodiversity loss detected."
            self.gemini_claims = [
                {"claim": "Sourcing only waste wood and residuals", "category": "Biomass", "verdict": "DISPUTED", "reasoning": "Satellite imagery shows harvesting of whole logs from primary growth zones."},
                {"claim": "Preservation of ancient woodland sites", "category": "Conservation", "verdict": "DISPUTED", "reasoning": "Imagery confirms removal of 400+ year old canopy in protected quadrant."},
            ]
            self.gemini_red_flags = [
                "Harvesting detected in 'No-Go' Primary Forest Zone",
                "Haul road density exceeds sustainable limits",
                "Thermal anomalies suggest on-site pile burning (undocumented)",
            ]
            self.gemini_risk = "CRITICAL"
            self.gemini_recommendations = ["Immediate suspension of sourcing license", "Launch full biodiversity impact assessment", "Report to regulatory bodies"]
        else:
            self.gemini_summary = "Veridian credits appear backed by 'phantom forests'. The reported coordinates correspond to arid scrubland with negligible carbon sequestration potential."
            self.gemini_claims = [
                {"claim": "800 hectares mature forest under management", "category": "Deforestation", "verdict": "DISPUTED", "reasoning": "Satellite shows sparse scrub/desert terrain, not forest."},
                {"claim": "Biodiversity index 0.87", "category": "Biodiversity", "verdict": "UNVERIFIABLE", "reasoning": "Landscape insufficient to support claimed distinct species count."},
            ]
            self.gemini_red_flags = [
                "Vegetation index (NDVI) < 0.2 (Barren)",
                "No infrastructure visible for forest management",
                "Land ownership records ambiguous",
            ]
            self.gemini_risk = "HIGH"
            self.gemini_recommendations = ["Reject credit listing", "Flag for fraud investigation"]


# ═══════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════

def glass_pane(child, danger_var, **kwargs):
    padding_val = kwargs.pop("padding", "2rem")
    base_style = {
        "background": "rgba(3, 15, 13, 0.8)",
        "backdrop_filter": "blur(20px)",
        "border": rx.cond(
            danger_var,
            "1px solid rgba(239, 68, 68, 0.7)",
            "1px solid rgba(94, 234, 212, 0.3)",
        ),
        "border_radius": "12px",
        "padding": padding_val,
        "position": "relative",
        "box_shadow": rx.cond(
            danger_var,
            "0 0 30px rgba(239, 68, 68, 0.25)",
            "none",
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

    button_base_style = {
        "height": "40px",
        "font_family": "monospace",
        "font_size": "11px",
        "letter_spacing": "0.15em",
        "background_color": "rgba(255, 255, 255, 0.03)",
        "backdrop_filter": "blur(5px)",
        "transition": "transform 0.2s",
        "cursor": "pointer",
        "_hover": {"transform": "scale(1.05)", "background_color": "rgba(255, 255, 255, 0.08)"},
        "_active": {"transform": "scale(0.95)"},
    }

    return rx.box(
        # ── Global CSS ──
        rx.el.style("""
            @keyframes scanline {
                0% { transform: translateY(-100%); }
                100% { transform: translateY(100%); }
            }
            @keyframes zoom-into-map {
                0% { transform: scale(1); filter: brightness(1); }
                60% { transform: scale(2); filter: brightness(0.6); }
                100% { transform: scale(3); filter: brightness(0.3); }
            }
            @keyframes spin-slow {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            @keyframes pulse-ring {
                0% { transform: scale(1); opacity: 0.6; }
                100% { transform: scale(1.8); opacity: 0; }
            }
            .scan-overlay {
                background: linear-gradient(to bottom, transparent, rgba(94, 234, 212, 0.08), transparent);
                animation: scanline 4s linear infinite;
            }
            .zoom-active iframe {
                animation: zoom-into-map 2s ease-in-out forwards;
            }
            .no-scrollbar::-webkit-scrollbar { display: none; }
            .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
            .heatmap-container {
                cursor: grab;
                overflow: hidden;
            }
            .heatmap-container:active {
                cursor: grabbing;
            }
        """),

        # ═══════════════════════════════════════════════════════
        # LAYER 0 — BACKGROUND IMAGE (z-index: 0)
        # ═══════════════════════════════════════════════════════
        rx.box(
            rx.image(
                src="/vantage_bg.png",
                width="100%",
                height="100%",
                fit="cover",
            ),
            style={
                "position": "fixed",
                "top": "0",
                "left": "0",
                "right": "0",
                "bottom": "0",
                "z_index": "0",
                "pointer_events": "none",
                "filter": rx.cond(
                    State.is_danger,
                    "sepia(0.3) saturate(2) hue-rotate(-30deg) brightness(0.4)",
                    "brightness(0.5)",
                ),
                "transition": "filter 0.5s ease",
            },
        ),
        # LAYER 1 — EDGE VIGNETTE (non-interactive)
        rx.box(
            style={
                "position": "fixed",
                "top": "0",
                "left": "0",
                "right": "0",
                "bottom": "0",
                "z_index": "1",
                "pointer_events": "none",
                "background": "radial-gradient(ellipse at center, transparent 30%, rgba(0,0,0,0.55) 100%)",
            },
        ),

        # ═══════════════════════════════════════════════════════
        # MAIN LAYOUT — sidebar + content
        # ═══════════════════════════════════════════════════════
        rx.flex(
            # ── SIDEBAR ──
            glass_pane(
                rx.vstack(
                    rx.heading(
                        "VANTAGE",
                        style={"color": accent_color},
                        class_name="tracking-[0.4em] text-2xl font-light mb-4",
                    ),
                    rx.box(class_name="w-full h-px bg-white/10 mb-4"),
                    rx.vstack(
                        rx.text(
                            "TARGETS",
                            class_name="text-[9px] text-white/30 tracking-[0.3em] mb-2",
                        ),
                        rx.button(
                            "Ford (Clean)",
                            on_click=State.set_company("Ford (Clean)", "0%"),
                            class_name="w-full h-14 justify-start px-6 bg-white/5 border-l-2 border-emerald-500/30 text-white/70 text-[11px] tracking-widest transition-all hover:border-emerald-400 hover:text-white hover:pl-8 rounded-r",
                        ),
                        rx.button(
                            "Drax (Factory)",
                            on_click=State.set_company("Drax (Factory)", "0%"),
                            class_name="w-full h-14 justify-start px-6 bg-white/5 border-l-2 border-emerald-500/30 text-white/70 text-[11px] tracking-widest transition-all hover:border-emerald-400 hover:text-white hover:pl-8 rounded-r",
                        ),
                        rx.button(
                            "Veridian (Ghost)",
                            on_click=State.set_company("Veridian (Ghost)", "0%"),
                            class_name="w-full h-14 justify-start px-6 bg-white/5 border-l-2 border-emerald-500/30 text-white/70 text-[11px] tracking-widest transition-all hover:border-emerald-400 hover:text-white hover:pl-8 rounded-r",
                        ),
                        spacing="3",
                        width="100%",
                    ),
                    rx.spacer(),
                    rx.box(class_name="w-full h-px bg-white/10 mb-4"),
                    rx.vstack(
                        rx.text(
                            "RISK INDEX",
                            style={"color": accent_color},
                            class_name="text-[9px] tracking-[0.3em] mb-1",
                        ),
                        rx.text(
                            State.hypocrisy_score,
                            style={"color": accent_color},
                            class_name="text-5xl font-black leading-none",
                        ),
                        rx.text(
                            State.current_company,
                            class_name="text-[9px] text-white/40 tracking-widest mt-2 uppercase",
                        ),
                        align_items="start",
                        width="100%",
                    ),
                    height="100%",
                ),
                danger_var=State.is_danger,
                width="320px",
                min_width="320px",
                height="calc(100vh - 3rem)",
                padding="1.5rem",
            ),

            # ── MAIN CONTENT AREA ──
            rx.vstack(
                # ── TOP BAR ──
                glass_pane(
                    rx.hstack(
                        rx.vstack(
                            rx.heading(
                                State.current_company,
                                style={"color": accent_color},
                                size="6",
                                class_name="font-bold",
                            ),
                            rx.text(
                                "Satellite AI Discrepancy Analysis",
                                class_name="font-light text-xs text-white/40",
                            ),
                            align_items="start",
                            spacing="1",
                        ),
                        rx.spacer(),
                        # Live feed indicator
                        rx.cond(
                            State.view_mode == "MAP",
                            rx.hstack(
                                rx.box(
                                    class_name="w-2 h-2 rounded-full bg-emerald-500 animate-pulse",
                                ),
                                rx.text(
                                    "LIVE FEED",
                                    class_name="text-[9px] text-emerald-400 tracking-widest font-mono",
                                ),
                                spacing="2",
                                class_name="border border-emerald-500/30 px-3 py-1.5 rounded-full mr-4",
                            ),
                        ),
                        rx.hstack(
                            rx.button(
                                "RESET",
                                on_click=State.reset_state,
                                style={
                                    **button_base_style,
                                    "border": "1px solid rgba(239, 68, 68, 0.4)",
                                    "color": primary_red,
                                },
                            ),
                            rx.button(
                                "DEEP SCAN",
                                on_click=State.run_deep_scan,
                                style={
                                    **button_base_style,
                                    "border": f"1px solid {primary_emerald}",
                                    "color": accent_color,
                                },
                            ),
                            spacing="3",
                        ),
                        width="100%",
                        align_items="center",
                    ),
                    danger_var=State.is_danger,
                    padding="1rem 1.5rem",
                    width="100%",
                ),

                # ── MAP / HEATMAP + LOGS ROW ──
                rx.hstack(
                    # Map/Heatmap Window (60%)
                    glass_pane(
                        rx.box(
                            # ── MAP VIEW (ALWAYS RENDERED, BOTTOM LAYER) ──
                            rx.box(
                                rx.html(
                                    f'<iframe width="100%" height="100%" frameborder="0" '
                                    f'style="border:0;display:block;" '
                                    f'src="{State.map_url}" allowfullscreen></iframe>',
                                    width="100%",
                                    height="100%",
                                ),
                                class_name=State.active_map_class,
                                style={
                                    "position": "absolute", "inset": "0",
                                    "width": "100%", "height": "100%",
                                    "border_radius": "8px", "overflow": "hidden",
                                    "z_index": "10",
                                    "opacity": rx.cond(State.view_mode == "MAP", "1", "0"),
                                    "transition": "opacity 0.8s ease-in-out",
                                },
                            ),
                            
                            # ── HEATMAP VIEW (ALWAYS RENDERED, TOP LAYER) ──
                            rx.box(
                                # Heatmap image (zoomable, draggable)
                                rx.box(
                                    rx.image(
                                        src=State.heatmap_url,
                                        class_name="w-full h-full object-cover select-none",
                                        draggable=False,
                                    ),
                                    style={
                                        "width": f"{State.heatmap_zoom_level * 100}%",
                                        "height": f"{State.heatmap_zoom_level * 100}%",
                                        "transition": "width 0.3s ease, height 0.3s ease",
                                    },
                                    id="heatmap-inner",
                                ),
                                # Zoom controls (bottom-right)
                                rx.vstack(
                                    rx.button(
                                        "+",
                                        on_click=State.zoom_heatmap_in,
                                        class_name="w-8 h-8 bg-black/80 text-white border border-white/20 hover:bg-white/20 text-sm font-mono rounded",
                                    ),
                                    rx.button(
                                        "-",
                                        on_click=State.zoom_heatmap_out,
                                        class_name="w-8 h-8 bg-black/80 text-white border border-white/20 hover:bg-white/20 text-sm font-mono rounded",
                                    ),
                                    spacing="1",
                                    style={
                                        "position": "absolute",
                                        "bottom": "1rem",
                                        "right": "1rem",
                                        "z_index": "40",
                                    },
                                ),
                                # Return button
                                rx.button(
                                    "RETURN TO LIVE FEED",
                                    on_click=State.toggle_view,
                                    class_name="absolute bottom-4 left-1/2 -translate-x-1/2 z-40 bg-black/80 text-white text-[11px] px-6 py-3 border border-emerald-500/40 hover:bg-emerald-500/20 tracking-widest font-mono transition-all cursor-pointer",
                                ),
                                # Drag-to-pan script
                                rx.el.script("""
                                    (function() {
                                        const container = document.getElementById('heatmap-container');
                                        const inner = document.getElementById('heatmap-inner');
                                        if (!container || !inner) return;
                                        let isDragging = false, startX, startY, scrollLeft, scrollTop;
                                        container.addEventListener('mousedown', (e) => {
                                            isDragging = true;
                                            startX = e.pageX - container.offsetLeft;
                                            startY = e.pageY - container.offsetTop;
                                            scrollLeft = container.scrollLeft;
                                            scrollTop = container.scrollTop;
                                            container.style.cursor = 'grabbing';
                                        });
                                        container.addEventListener('mouseleave', () => {
                                            isDragging = false;
                                            container.style.cursor = 'grab';
                                        });
                                        container.addEventListener('mouseup', () => {
                                            isDragging = false;
                                            container.style.cursor = 'grab';
                                        });
                                        container.addEventListener('mousemove', (e) => {
                                            if (!isDragging) return;
                                            e.preventDefault();
                                            const x = e.pageX - container.offsetLeft;
                                            const y = e.pageY - container.offsetTop;
                                            container.scrollLeft = scrollLeft - (x - startX);
                                            container.scrollTop = scrollTop - (y - startY);
                                        });
                                    })();
                                """),
                                class_name="heatmap-container",
                                id="heatmap-container",
                                style={
                                    "position": "absolute",
                                    "inset": "0",  # Covers the parent
                                    "width": "100%",
                                    "height": "100%",
                                    "overflow": "hidden",
                                    "background": "black",
                                    "border_radius": "8px",
                                    "cursor": "grab",
                                    "z_index": "20",
                                    "opacity": rx.cond(State.view_mode == "HEATMAP", "1", "0"),
                                    "pointer_events": rx.cond(State.view_mode == "HEATMAP", "auto", "none"),
                                    "transition": "opacity 0.8s ease-in-out",
                                },
                            ),
                            
                            style={
                                "width": "100%",
                                "height": "100%",
                                "position": "relative",
                            },
                        ),
                        danger_var=State.is_danger,
                        width="60%",
                        padding="0px",
                        style={"height": "100%", "overflow": "hidden"},
                    ),

                    # System Logs (40%)
                    glass_pane(
                        rx.box(
                            corner_brackets(danger_var=State.is_danger),
                            rx.text(
                                "SYSTEM LOGS",
                                style={"color": accent_color},
                                class_name="font-bold tracking-[0.4em] text-[9px] mb-3 mt-6",
                            ),
                            rx.scroll_area(
                                rx.vstack(
                                    rx.foreach(
                                        State.logs,
                                        lambda log: rx.text(
                                            f"> {log}",
                                            style={"color": accent_color},
                                            class_name="font-mono text-[9px] uppercase opacity-70",
                                        ),
                                    ),
                                    align_items="start",
                                ),
                                height="100%",
                                class_name="no-scrollbar",
                            ),
                            height="100%",
                        ),
                        danger_var=State.is_danger,
                        width="40%",
                        padding="1rem",
                        style={"height": "100%"},
                    ),
                    spacing="4",
                    width="100%",
                    height="500px",
                    align_items="stretch",
                ),

                # ── DOCUMENT NEURAL LINK (full width below map) ──
                glass_pane(
                    rx.box(
                        rx.hstack(
                            rx.icon(tag="file-text", size=14, color=accent_color),
                            rx.text(
                                "Document Neural Link",
                                class_name="font-bold text-white text-[10px] tracking-wider",
                            ),
                            spacing="2",
                        ),
                        rx.box(class_name="w-full h-px bg-white/10 my-3"),
                        rx.cond(
                            State.show_report,
                            # ── RICH REPORT VIEW ──
                            rx.vstack(
                                # Risk badge header
                                rx.hstack(
                                    rx.box(
                                        rx.text(
                                            State.gemini_risk,
                                            class_name="font-mono font-bold text-[11px] tracking-widest",
                                        ),
                                        class_name=rx.cond(
                                            State.gemini_risk == "CRITICAL",
                                            "bg-red-500/20 text-red-400 border border-red-500/50 px-4 py-1.5 rounded-full",
                                            rx.cond(
                                                State.gemini_risk == "HIGH",
                                                "bg-orange-500/20 text-orange-400 border border-orange-500/50 px-4 py-1.5 rounded-full",
                                                "bg-emerald-500/20 text-emerald-400 border border-emerald-500/50 px-4 py-1.5 rounded-full",
                                            ),
                                        ),
                                    ),
                                    rx.text(
                                        "DOCUMENT ANALYSIS COMPLETE",
                                        class_name="text-white/40 text-[9px] tracking-widest font-mono",
                                    ),
                                    spacing="3",
                                    align_items="center",
                                ),
                                # Executive summary
                                rx.box(
                                    rx.text(
                                        State.gemini_summary,
                                        class_name="text-white/80 text-[11px] leading-relaxed font-mono",
                                    ),
                                    class_name="bg-white/5 p-4 rounded-lg border-l-2 border-white/20 w-full",
                                ),
                                # Red flags (if any)
                                rx.cond(
                                    State.gemini_red_flags.length() > 0,
                                    rx.box(
                                        rx.hstack(
                                            rx.icon(tag="triangle-alert", size=12, color="#ef4444"),
                                            rx.text(
                                                "RED FLAGS DETECTED",
                                                class_name="text-red-400 text-[9px] font-bold tracking-widest",
                                            ),
                                            spacing="2",
                                            class_name="mb-2",
                                        ),
                                        rx.vstack(
                                            rx.foreach(
                                                State.gemini_red_flags,
                                                lambda flag: rx.hstack(
                                                    rx.text("!", class_name="text-red-400 font-bold text-[10px] w-4 flex-shrink-0"),
                                                    rx.text(
                                                        flag,
                                                        class_name="text-red-300/80 text-[10px] font-mono",
                                                    ),
                                                    spacing="2",
                                                    align_items="start",
                                                ),
                                            ),
                                            spacing="1",
                                        ),
                                        class_name="bg-red-500/10 border border-red-500/30 p-4 rounded-lg w-full",
                                    ),
                                ),
                                # Claims list (scrollable)
                                rx.box(
                                    rx.text(
                                        "EXTRACTED CLAIMS",
                                        class_name="text-white/40 text-[9px] tracking-widest font-mono mb-2",
                                    ),
                                    rx.scroll_area(
                                        rx.vstack(
                                            rx.foreach(
                                                State.gemini_claims,
                                                lambda claim: rx.box(
                                                    rx.hstack(
                                                        rx.box(
                                                            rx.text(
                                                                claim["verdict"],
                                                                class_name="text-[8px] font-bold tracking-wider",
                                                            ),
                                                            class_name=rx.cond(
                                                                claim["verdict"] == "VERIFIED",
                                                                "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 px-2 py-0.5 rounded text-center min-w-[80px]",
                                                                rx.cond(
                                                                    claim["verdict"] == "DISPUTED",
                                                                    "bg-red-500/20 text-red-400 border border-red-500/40 px-2 py-0.5 rounded text-center min-w-[80px]",
                                                                    "bg-yellow-500/20 text-yellow-400 border border-yellow-500/40 px-2 py-0.5 rounded text-center min-w-[80px]",
                                                                ),
                                                            ),
                                                        ),
                                                        rx.vstack(
                                                            rx.text(
                                                                claim["claim"],
                                                                class_name="text-white/80 text-[10px] font-mono",
                                                            ),
                                                            rx.text(
                                                                claim["reasoning"],
                                                                class_name="text-white/40 text-[9px] font-mono",
                                                            ),
                                                            spacing="1",
                                                            align_items="start",
                                                        ),
                                                        spacing="3",
                                                        align_items="start",
                                                        width="100%",
                                                    ),
                                                    class_name="bg-white/5 p-3 rounded border-l border-white/10 w-full",
                                                ),
                                            ),
                                            spacing="2",
                                            width="100%",
                                        ),
                                        max_height="200px",
                                        class_name="no-scrollbar",
                                    ),
                                    width="100%",
                                ),
                                spacing="4",
                                width="100%",
                            ),
                            # ── UPLOAD VIEW ──
                            rx.box(
                                # Loading overlay (absolute within this panel)
                                rx.cond(
                                    State.is_analyzing,
                                    rx.box(
                                        rx.vstack(
                                            rx.box(
                                                rx.box(
                                                    class_name="w-12 h-12 border-2 border-emerald-500/30 border-t-emerald-400 rounded-full",
                                                    style={"animation": "spin-slow 1.5s linear infinite"},
                                                ),
                                                rx.box(
                                                    class_name="absolute inset-0 w-12 h-12 border-2 border-transparent border-t-emerald-300/50 rounded-full",
                                                    style={"animation": "pulse-ring 2s ease-out infinite"},
                                                ),
                                                style={"position": "relative"},
                                            ),
                                            rx.text(
                                                "ANALYZING DOCUMENT...",
                                                class_name="text-emerald-400 text-[11px] font-mono tracking-widest mt-4 animate-pulse",
                                            ),
                                            rx.text(
                                                "Cross-referencing satellite intelligence",
                                                class_name="text-white/30 text-[9px] font-mono tracking-wider",
                                            ),
                                            align_items="center",
                                            justify_content="center",
                                        ),
                                        style={
                                            "position": "absolute",
                                            "inset": "0",
                                            "background": "rgba(0, 0, 0, 0.85)",
                                            "z_index": "30",
                                            "display": "flex",
                                            "align_items": "center",
                                            "justify_content": "center",
                                            "border_radius": "12px",
                                        },
                                    ),
                                ),
                                rx.vstack(
                                    rx.upload(
                                        rx.vstack(
                                            rx.button(
                                                "SELECT REPORT",
                                                class_name="bg-white/10 text-white text-[9px] px-3 py-1.5 rounded hover:bg-white/20",
                                            ),
                                            rx.text(
                                                "Drag & Drop PDF/CSV",
                                                class_name="text-white/30 text-[8px] mt-1",
                                            ),
                                            align_items="center",
                                            padding="1.5rem",
                                            border="1px dashed rgba(255,255,255,0.2)",
                                            border_radius="8px",
                                        ),
                                        id="report_upload",
                                        multiple=False,
                                        accept={
                                            "application/pdf": [".pdf"],
                                            "image/*": [".png", ".jpg"],
                                            ".json": [".json"],
                                            ".csv": [".csv"],
                                        },
                                        max_files=1,
                                        border="1px dotted rgba(255,255,255,0.1)",
                                        padding="0.5em",
                                    ),
                                    rx.button(
                                        "ANALYZE EVIDENCE",
                                        on_click=State.process_upload(
                                            rx.upload_files(upload_id="report_upload")
                                        ),
                                        class_name="w-full bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/40 py-1.5 rounded text-[9px] tracking-widest mt-1",
                                    ),
                                    width="100%",
                                ),
                                style={"position": "relative"},
                                width="100%",
                            ),
                        ),
                        width="100%",
                    ),
                    danger_var=State.is_danger,
                    width="100%",
                    padding="1rem",
                ),

                spacing="4",
                width="100%",
                overflow_y="auto",
                flex="1",
                class_name="no-scrollbar",
                padding_bottom="2rem",
            ),

            gap="1.5rem",
            padding="1.5rem",
            width="100%",
            height="100vh",
            align_items="stretch",
            style={
                "position": "relative",
                "z_index": "10",
                "pointer_events": "auto",
            },
        ),

        # Root container styling
        style={
            "width": "100vw",
            "height": "100vh",
            "overflow": "hidden",
            "background": "transparent",
        },
    )

app = rx.App()
app.add_page(index)
