import reflex as rx

config = rx.Config(
    app_name="VANTAGE_CXC_Hackathon",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)