import reflex as rx

def hypocrisy_card(score, label: str):
    # 1. Handle the 'score' display text correctly for both int and Var
    # If it's a Var, we use .to_string(), otherwise we just use the int
    score_percent = (
        score.to_string() + "%" 
        if isinstance(score, rx.Var) 
        else f"{score}%"
    )

    # 2. Use rx.cond for the color (handles both types automatically)
    color = rx.cond(
        score < 30, 
        "jade", 
        rx.cond(score < 70, "amber", "red")
    )
    
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(label, class_name="text-slate-400 font-medium text-sm"),
                rx.spacer(),
                # Use our smart score_percent here
                rx.badge(score_percent, color_scheme=color, variant="soft"),
                width="100%"
            ),
            rx.box(
                rx.box(
                    # Handle width correctly for both types
                    width=score.to_string() + "%" if isinstance(score, rx.Var) else f"{score}%",
                    background_color=rx.cond(
                        score < 30, 
                        "var(--jade-9)", 
                        rx.cond(score < 70, "var(--amber-9)", "var(--red-9)")
                    ),
                    class_name="h-full transition-all duration-1000 rounded-full",
                ),
                class_name="w-full h-1.5 bg-slate-800 rounded-full mt-2"
            ),
            align_items="start",
        ),
        class_name="p-5 bg-slate-900/40 border border-slate-800 rounded-xl w-full"
    )