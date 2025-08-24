# from alert_service import send_discord_message, send_table, send_timeseries, send_news

# # Example: Simple text
# send_discord_message("Hello from Python 👋")

# # Example: Table
# send_table(
#     headers=["ID", "Name", "Score"],
#     rows=[
#         [1, "Alice", 90],
#         [2, "Bob", 85],
#         [3, "Charlie", 92]
#     ],
#     title="Leaderboard"
# )

# # Example: Time series
# send_timeseries(
#     {
#         "2025-08-09T10:00": 100,
#         "2025-08-09T11:00": 105,
#         "2025-08-09T12:00": 98
#     },
#     title="Hourly Values"
# )

# # Example: News alert
# send_news(
#     headline="New Feature Released 🚀",
#     body="We just rolled out a new dashboard for analytics users.",
#     source="Company Blog"
# )


import pandas as pd
from alert_service import send_dataframe_to_discord

# Example DataFrame
data = {
    "ID": [1, 2, 3, 4, 5],
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "Score": [95, 87, 92, 88, 100]
}
df = pd.DataFrame(data)

# Send table to Discord
send_dataframe_to_discord(df, title="Leaderboard")
