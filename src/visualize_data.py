"""
title: Visualize data
author: Moicky
author_url: https://github.com/Moicky/open-webui-functions
icon_url: data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iODAgLTg4MCA4ODAgODgwIiB3aWR0aD0iMjRweCIgZmlsbD0iIzlCOUI5QiI+PHBhdGggZD0iTTEyMC0xMjB2LTgwbDgwLTgwdjE2MGgtODBabTE2MCAwdi0yNDBsODAtODB2MzIwaC04MFptMTYwIDB2LTMyMGw4MCA4MXYyMzloLTgwWm0xNjAgMHYtMjM5bDgwLTgwdjMxOWgtODBabTE2MCAwdi00MDBsODAtODB2NDgwaC04MFpNMTIwLTMyN3YtMTEzbDI4MC0yODAgMTYwIDE2MCAyODAtMjgwdjExM0w1NjAtNDQ3IDQwMC02MDcgMTIwLTMyN1oiLz48L3N2Zz4=
version: 0.1.0
required_open_webui_version: 0.3.17
license: MIT
"""

from typing import Optional, Any, Callable, Awaitable
from open_webui.utils.misc import get_last_assistant_message_item
from pydantic import BaseModel, Field
import traceback
import requests
import logging
import json
import sys


SYSTEM_PROMPT_BUILD_CHARTS = """

Objective:
Your goal is to read the query, extract the data, choose the appropriate chart to present the data, and produce the HTML to display it.

Steps:

	1.	Read and Examine the Query:
	•	Understand the user’s question and identify the data provided.
	2.	Analyze the Data:
	•	Examine the data in the query to determine the appropriate chart type (e.g., bar chart, pie chart, line chart) for effective visualization.
	3.	Generate HTML:
	•	Create the HTML code to present the data using the selected chart format.
	4.	Handle No Data Situations:
	•	If there is no data in the query or the data cannot be presented as a chart, generate a humorous or funny HTML response indicating that the data cannot be presented.
    5.	Calibrate the chart scale based on the data:
	•	based on the data try to make the scale of the chart as readable as possible.

Key Considerations:

	-	Your output should only include HTML code, without any additional text.
    -   Generate only HTML. Do not include any additional words or explanations.
    -   Make to remove any character other non alpha numeric from the data.
    -   is the generated HTML Calibrate the chart scale based on the data for eveything to be readable.
    -   Generate only html code , nothing else , only html.


Example1 : 
'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Chart</title>
    <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
</head>
<body style="min-height: 600px;">
    <div id="chart" style="width: 100%; height: 95vh;"></div>
    <script>
        // Data for the chart
        var data = [{
            x: [''Category 1'', ''Category 2'', ''Category 3''],
            y: [20, 14, 23],
            type: ''bar''
        }];

        // Layout for the chart
        var layout = {
            title: ''Interactive Bar Chart'',
            xaxis: {
                title: ''Categories''
            },
            yaxis: {
                title: ''Values''
            },
        };

        // Render the chart
        Plotly.newPlot(''chart'', data, layout);

        // Function to update chart attributes
        function updateChartAttributes(newData, newLayout) {
            Plotly.react(''chart'', newData, newLayout);
        }

        // Example of updating chart attributes
        var newData = [{
            x: [''New Category 1'', ''New Category 2'', ''New Category 3''],
            y: [10, 22, 30],
            type: ''bar''
        }];

        var newLayout = {
            title: ''Updated Bar Chart'',
            xaxis: {
                title: ''New Categories''
            },
            yaxis: {
                title: ''New Values''
            },
        };

        // Call updateChartAttributes with new data and layout
        // updateChartAttributes(newData, newLayout);
    </script>
</body>
</html>
'''

2.	No Data or Unchartable Data:
''' 
<html>
<body>
    <h1>We''re sorry, but your data can''t be charted.</h1>
    <p>Maybe try feeding it some coffee first?</p>
    <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3MzdjYyeXUweWZqd2ZrNm95NGp0eGQwYWhoeGE1YTNtaXRhNzQ2ayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ji6zzUZwNIuLS/giphy.gif" alt="Confused GIF">
</body>
</html>

'''

"""
USER_PROMPT_GENERATE_HTML = """
Giving this query  {Query} generate the necessary html.
"""

USAGE_TRACKING_MODULE = "function_open_router"


class Action:
    class Valves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        OPENAI_KEY: str = Field(
            default="",
            description="key to consume OpenAI interface like LLM for example a litellm key.",
        )
        OPENAI_BASE_URL: str = Field(
            default="",
            description="Host where to consume the OpenAI interface like llm",
        )
        MODEL: str = Field(
            default="gpt-4o-mini",
            description="Model to use for the action",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.openai = None
        self.logger = logging.getLogger("VISUAL  ")
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        self.logger.handlers = [handler]
        self.logger.setLevel(logging.INFO)

    async def emit(
        self,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        description: str,
        done: Optional[bool] = False,
    ):
        if self.valves.show_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": description, "done": done},
                }
            )

    async def action(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_call__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Optional[dict]:
        self.logger.info(f"action:{__name__} started")
        await self.emit(__event_emitter__, "Analysing Data", False)

        if USAGE_TRACKING_MODULE not in sys.modules:
            self.logger.error(f"Module {USAGE_TRACKING_MODULE} is not loaded")
        else:
            usage_tracking_module = sys.modules[USAGE_TRACKING_MODULE]
            usage_persistence_manager = usage_tracking_module.UsagePersistenceManager()

        try:
            message = get_last_assistant_message_item(body["messages"])
            message_content = message["content"]

            self.logger.info(f"Generating HTML for the data with {self.valves.MODEL}")
            resp = requests.post(
                self.valves.OPENAI_BASE_URL + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.valves.OPENAI_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.valves.MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_BUILD_CHARTS},
                        {
                            "role": "user",
                            "content": USER_PROMPT_GENERATE_HTML.format(
                                Query=message_content
                            ),
                        },
                    ],
                    "max_tokens": 10000,
                    "n": 1,
                    "stop": None,
                    "temperature": 0.7,
                    "usage": {"include": True},
                    "stream": False,
                },
            )

            resp.raise_for_status()

            result = resp.json()
            html_content = result["choices"][0]["message"]["content"]

            if usage_persistence_manager:
                user = __user__[0] if type(__user__) == tuple else __user__
                await usage_persistence_manager.log_usage_fact(
                    user_id=user["id"],
                    model=self.valves.MODEL,
                    metadata=json.dumps(
                        {
                            "chat_id": body["chat_id"],
                            "session_id": body["session_id"],
                        }
                    ),
                    input_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                    output_tokens=result.get("usage", {}).get("completion_tokens", 0),
                    total_cost=result.get("usage", {}).get("cost", 0),
                )

            text = f"\n\nYou would use this to visualize the data:\n```html\n{html_content}\n```"

            await self.emit(__event_emitter__, "Visualise the chart", True)
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": text},
                }
            )

            self.logger.info(f"objects visualized")
        except Exception as e:
            error_message = f"Error visualizing JSON: {str(e)}"
            self.logger.error(f"Error: {error_message}")
            self.logger.error(traceback.format_exc())
            message["content"] += f"\n\nError: {error_message}"

            await self.emit(__event_emitter__, "Error Visualizing JSON", True)

        self.logger.info(f"action:{__name__} completed")
