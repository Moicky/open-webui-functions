"""
title: OpenRouter Usage Bot
author: Moicky
author_url: https://github.com/Moicky/open-webui-functions
version: 0.1.0
required_open_webui_version: 0.3.17
license: MIT
"""

from open_webui.utils.misc import get_messages_content, get_last_user_message
from open_webui.internal.db import get_db, engine
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from datetime import datetime
from typing import Optional
from sqlalchemy import text
import pandas as pd
import requests
import logging
import sys
import re


class Pipe:
    class Valves(BaseModel):
        SUPERUSERS: str = Field(
            default="",
            description="Comma-separated list of user emails with elevated admin-like bot capabilities",
        )

        OPEN_ROUTER_API_KEY: str = Field(
            default="",
            description="Open Router API key",
        )
        SQL_ASSISTANT_MODEL: str = Field(
            default="anthropic/claude-3.7-sonnet",
            description="Model to use for SQL generation from natural language questions",
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "usage-bot"
        self.name = "usage-bot"

        self.valves = self.Valves()

        self.logger = logging.getLogger("OpenRouterTracker")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.handlers = [handler]
        self.logger.setLevel(logging.INFO)

    def get_provider_models(self):
        return [{"id": "admin.usage-bot", "name": "usage-bot"}]

    def is_superuser(self, __user__: dict):
        if __user__["role"] == "admin":
            return True

        return __user__["email"] in [
            user.strip() for user in self.valves.SUPERUSERS.split(",")
        ]

    def strip_markdown_formatting(self, text: str) -> str:
        """
        Strip markdown formatting from text, particularly bold (**) markers.
        """
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        return text.strip()

    async def pipe(self, body: dict, __user__: dict) -> str:
        command = get_last_user_message(body["messages"]).strip()
        command = self.strip_markdown_formatting(command)

        if command.startswith("/usage_costs"):
            command = "/usage_stats" + command[len("/usage_costs") :]

        if command == "/help":
            return self.print_help(__user__)
        else:
            return await self.handle_command(__user__, body, command)

    async def handle_command(self, __user__, body, command):
        if command == "/balance":
            return self.get_balance()

        if match := re.match(r"/usage_stats\s+all(?:\s+(\d+)d)?", command):
            days = int(match.group(1)) if match.group(1) else 30

            return self.generate_all_users_report(days)

        if match := re.match(r"/usage_stats\s+([^\s]+@[^\s]+)(?:\s+(\d+)d)?", command):
            specific_user = match.group(1)
            days = int(match.group(2)) if match.group(2) else 30

            return self.generate_single_user_report(days, specific_user)

        if match := re.match(r"/usage_stats(?:\s+(\d+)d)?", command):
            days = int(match.group(1)) if match.group(1) else 30

            return self.generate_single_user_report(
                days, __user__["id"], __user__["email"]
            )

        if command.startswith("/run_sql "):
            if self.is_superuser(__user__):
                return self.run_sql_command(command[len("/run_sql ") :])
            else:
                return "Sorry, this feature is only available to Admins"

        if command.startswith("/ask "):
            if self.is_superuser(__user__):
                return await self.handle_ask_command(__user__, body, command[5:])
            else:
                return "Sorry, this feature is only available to Admins"

        return "Invalid command\n\n" + self.print_help(__user__)

    def print_help(self, __user__):
        help_message = (
            "**Available Commands**\n"
            "* **/balance** Check current API balance\n"
            "* **/usage_stats all 45d** or **/usage_costs all 45d** stats by all users for 45 days\n"
            "* **/usage_stats 30d** or **/usage_costs 30d** my own usage stats for 30 days\n\n"
        )

        if self.is_superuser(__user__):
            help_message += (
                "**Available Commands (Admins Only)**\n"
                "* **/run_sql SELECT count(*) from usage_costs;** allows an admin to run arbitrary SQL SELECT from the database.\n  - For SQLite: use /run_sql PRAGMA table_info(usage_costs) to see available table columns\n  - For Postgres db: /run_sql SELECT * FROM information_schema.columns WHERE table_name = 'usage_costs'\n"
                "* **/ask** Ask questions about usage in natural language. SQL will be generated automatically.\n"
            )

        return help_message

    def get_usage_stats(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        """
        Retrieve total costs by user, summarized per user, model, currency, and date.

        :param user_id: Optional user id for filtering results
        :param start_date: Optional start date for filtering results
        :param end_date: Optional end date for filtering results
        :return: List of dictionaries containing summarized cost data
        """

        is_sqlite = "sqlite" in engine.url.drivername

        date_function = (
            "strftime('%Y-%m-%d', timestamp)"
            if is_sqlite
            else "to_char(timestamp, 'YYYY-MM-DD')"
        )

        query = f"""
            SELECT 
                u.email as user_email,
                uc.user_id,
                uc.model,
                {date_function} as date,
                SUM(uc.total_cost) as total_cost,
                SUM(uc.input_tokens) as total_input_tokens,
                SUM(uc.output_tokens) as total_output_tokens,
                COUNT(uc.id) as messages_count
            FROM usage_costs uc
            JOIN user u ON uc.user_id = u.id
            {{where_clause}}
            GROUP BY uc.user_id, uc.model, {date_function}
            ORDER BY uc.user_id, {date_function}, uc.model
            """

        where_conditions = []
        params = {}

        if user_id:
            where_conditions.append("user_id = :user_id")
            params["user_id"] = user_id

        if start_date:
            where_conditions.append("timestamp >= :start_date")
            params["start_date"] = start_date

        if end_date:
            # Include the entire end_date by setting it to the start of the next day
            next_day = end_date + timedelta(days=1)
            where_conditions.append("timestamp < :end_date")
            params["end_date"] = next_day

        where_clause = (
            "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        )
        query = query.format(where_clause=where_clause)

        try:
            with get_db() as db:
                result = db.execute(text(query), params)
                rows = result.fetchall()

                summary = [
                    {
                        "user_id": row.user_id,
                        "user_email": row.user_email,
                        "model": row.model,
                        "date": row.date,
                        "total_cost": float(row.total_cost),
                        "total_input_tokens": row.total_input_tokens,
                        "total_output_tokens": row.total_output_tokens,
                        "messages_count": row.messages_count,
                    }
                    for row in rows
                ]

                return summary

        except Exception as e:
            self.logger.error(e)
            raise

    def generate_all_users_report(self, days: int):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get usage stats for all users
        stats = self.get_usage_stats(start_date=start_date, end_date=end_date)

        if not stats:
            return f"No usage data found in the last {days} days."

        df = pd.DataFrame(stats)
        # Prepare the report
        report = f"## Usage Report for All Users\n"
        report += f"### Period: {start_date.date()} to {end_date.date()}\n\n"

        # Total costs by currency
        report += "#### Total Usage Costs:\n"
        report += f"- **{df.total_cost.sum():,.2f} $**\n"

        # Add total tokens information
        total_input_tokens = df["total_input_tokens"].sum()
        total_output_tokens = df["total_output_tokens"].sum()
        report += "\n#### Total Tokens Used:\n"
        report += f"- Input tokens:  **{total_input_tokens:,}**\n"
        report += f"- Output tokens: **{total_output_tokens:,}**\n"

        # Top 5 models by cost
        report += "\n#### Top 10 Models by Cost:\n"

        # Group by model and currency, then sum costs
        model_costs = df.groupby(["model"])[["total_cost"]].sum().reset_index()
        top_models = model_costs.nlargest(10, "total_cost")

        headers = ["Model", "Cost ($)"]
        rows = []
        for _, row in top_models.iterrows():
            model_name = row["model"]
            cost = row["total_cost"]
            rows.append([model_name, f"${cost:,.2f}" if cost > 0 else ""])

        # Render the table
        col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

        table = "```\n"  # Start code block for fixed-width formatting
        table += (
            " | ".join(
                f"{header:<{width}}" for header, width in zip(headers, col_widths)
            )
            + "\n"
        )
        table += "-|-".join("-" * width for width in col_widths) + "\n"
        for row in rows:
            table += (
                " | ".join(f"{cell:<{width}}" for cell, width in zip(row, col_widths))
                + "\n"
            )
        table += "```\n"  # End code block

        report += table

        # Top 20 users by cost
        report += "\n#### Top 20 Users by Cost:\n"

        # Get user totals and select top 20 users
        agg_dict = {}
        agg_dict["total_cost"] = "sum"
        agg_dict["total_input_tokens"] = "sum"
        agg_dict["total_output_tokens"] = "sum"
        agg_dict["messages_count"] = "sum"
        agg_dict["user_email"] = "first"

        user_totals = df.groupby("user_id").agg(agg_dict).round(2)
        top_users = user_totals.nlargest(20, "total_cost")

        headers = ["User", "Cost ($)", "Input Tokens", "Output Tokens", "Messages"]

        rows = []
        for _, row in top_users.iterrows():
            row_data = [row["user_email"]]
            row_data.extend(
                [
                    f"${row['total_cost']:,.2f}" if row["total_cost"] > 0 else "",
                    f"{row['total_input_tokens']:,}",
                    f"{row['total_output_tokens']:,}",
                    f"{row['messages_count']:,}",
                ]
            )
            rows.append(row_data)

        # Render the table
        col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

        table = "```\n"  # Start code block for fixed-width formatting
        table += (
            " | ".join(
                f"{header:<{width}}" for header, width in zip(headers, col_widths)
            )
            + "\n"
        )
        table += "-|-".join("-" * width for width in col_widths) + "\n"
        for row in rows:
            table += (
                " | ".join(f"{cell:<{width}}" for cell, width in zip(row, col_widths))
                + "\n"
            )
        table += "```\n"  # End code block

        report += table

        return report

    def generate_single_user_report(self, days: int, user_id: str, user_email: str):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get usage stats
        stats = self.get_usage_stats(
            user_id=user_id, start_date=start_date, end_date=end_date
        )

        if not stats:
            return f"No usage data found for user {user_id} in the last {days} days."

        # Convert to DataFrame for easy manipulation
        df = pd.DataFrame(stats)

        total = df["total_cost"].sum().round(2)

        # Prepare the report
        report = [
            f"## Usage Report for {user_email}",
            f"### Period: {start_date.date()} to {end_date.date()}",
            "",
            "#### Usage Costs:",
            "",
        ]

        report.append(f"- **{total:,.2f} $**")

        # Add total tokens information
        total_input_tokens = df["total_input_tokens"].sum()
        total_output_tokens = df["total_output_tokens"].sum()
        report.extend(
            [
                "",
                "#### Total Tokens Used:",
                f"- Input tokens:  **{total_input_tokens:,}**",
                f"- Output tokens: **{total_output_tokens:,}**",
                "",
            ]
        )

        # TOP 5 MODELS BY COST
        report.append("#### Top 5 Models by Cost:")
        report.append("")

        # Group data and select top 5 models by USD cost (currency-converted)
        model_costs = df.groupby(["model"])[["total_cost"]].sum().reset_index()

        top_models = model_costs.groupby("model")["total_cost"].sum().nlargest(5).index

        top_model_data = model_costs[model_costs["model"].isin(top_models)]

        # Prepare data for table rendering
        headers = ["Model", "$ (Cost)"]

        rows = []
        for model in top_models:
            row_data = [model]
            model_data = top_model_data[(top_model_data["model"] == model)]
            if len(model_data) > 0:
                cost = model_data["total_cost"].sum()

                row_data.append(f"${cost:,.2f}" if cost > 0 else "")
            else:
                row_data.append("")
            rows.append(row_data)

        # Render an ASCII table
        if rows:  # Only add table if we have data
            col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

            table_lines = [
                "```",
                " | ".join(
                    f"{header:<{width}}" for header, width in zip(headers, col_widths)
                ),
                "-|-".join("-" * width for width in col_widths),
            ]

            for row in rows:
                table_lines.append(
                    " | ".join(
                        f"{cell:<{width}}" for cell, width in zip(row, col_widths)
                    )
                )

            table_lines.append("```")
            report.extend(table_lines)

        return "\n".join(report)

    def run_sql_command(self, sql_query):
        # Sanitize the query
        sql_query = sql_query.strip()
        self.logger.info(f"SQL QUERY: {sql_query}")

        if not re.match(r"^(SELECT|PRAGMA TABLE_)", sql_query, re.IGNORECASE):
            err_msg = "Error: Query must start with SELECT or PRAGMA TABLE_LIST() or PRAGMA TABLE_INFO(table)"
            self.logger.error(f"run_sql | {err_msg}")
            return f"{err_msg}"

        if not sql_query.endswith(";"):
            sql_query += ";"

        if sql_query.count(";") > 1:

            err_msg = "Error: Query must not contain multiple semicolons (;)"
            self.logger.error(f"run_sql |  {err_msg}")
            return f"{err_msg}"

        try:
            with get_db() as db:
                result = db.execute(text(sql_query))

                # Check if the query returns rows before trying to fetch
                if result.returns_rows:
                    rows = result.fetchall()

                    if not rows:
                        msg = "Query executed successfully, but returned no results."
                        self.logger.info(f"run_sql |  {msg}")
                        return f"{msg}"

                    # Get column names
                    if hasattr(result, "keys"):
                        headers = result.keys()
                    else:  # Fallback for older SQLAlchemy versions or specific drivers
                        headers = (
                            rows[0]._fields
                            if rows and hasattr(rows[0], "_fields")
                            else []
                        )

                    # Format data
                    formatted_data = []
                    for row in rows:
                        formatted_row = []
                        for col in headers:
                            value = getattr(row, col, "")
                            # Add thousands separator for numeric values
                            if isinstance(value, (int, float)) and not isinstance(
                                value, bool
                            ):
                                if isinstance(value, int):
                                    formatted_value = f"{value:,}"
                                else:
                                    # For floats, keep decimal precision
                                    formatted_value = (
                                        f"{value:,.2f}"  # Keep .2f precision
                                    )
                            else:
                                formatted_value = str(value)
                            formatted_row.append(formatted_value)
                        formatted_data.append(formatted_row)

                    # Calculate column widths
                    col_widths = (
                        [
                            max(len(str(x)) for x in col)
                            for col in zip(headers, *formatted_data)
                        ]
                        if headers
                        else []
                    )

                    # Create a markdown table
                    table = "```\n"  # Start code block for fixed-width formatting
                    if headers:
                        table += (
                            " | ".join(
                                f"{header:<{width}}"
                                for header, width in zip(headers, col_widths)
                            )
                            + "\n"
                        )
                        table += "-|-".join("-" * width for width in col_widths) + "\n"
                    else:
                        table += "(No column headers returned)\n"

                    for row in formatted_data:
                        if headers:
                            table += (
                                " | ".join(
                                    f"{cell:<{width}}"
                                    for cell, width in zip(row, col_widths)
                                )
                                + "\n"
                            )
                        else:
                            table += (
                                str(row) + "\n"
                            )  # Simple representation if no headers

                    table += "```\n"  # End code block

                    # Add truncation notice if necessary
                    self.logger.info(
                        f"run_sql | returned query results {len(rows)} rows"
                    )

                    return f"Query results:\n\n{table}"

                else:
                    # For non-SELECT statements (UPDATE, INSERT, DELETE etc.) or SELECTs that return no rows by design
                    db.commit()  # Make sure to commit changes for DML
                    msg = "Query executed successfully. No rows returned (this is expected for non-SELECT queries or queries with no matching results)."
                    self.logger.info(f"run_sql | {msg}")
                    return msg

        except Exception as e:
            _, _, tb = sys.exc_info()
            line_number = tb.tb_lineno
            self.logger.error(f"run_sql | Error on line {line_number}: {e}")
            # Optionally, try to rollback in case of error during DML
            try:
                with get_db() as db_rollback:
                    db_rollback.rollback()
            except Exception as rollback_e:
                self.logger.error(f"run_sql | Error during rollback: {rollback_e}")
            return f"Error executing query: {e}"

    def get_balance(self) -> str:
        results = []
        errors = []

        if self.valves.OPEN_ROUTER_API_KEY:
            try:
                headers = {"Authorization": f"Bearer {self.valves.OPEN_ROUTER_API_KEY}"}
                response = requests.get(
                    "https://openrouter.ai/api/v1/credits", headers=headers
                )
                response.raise_for_status()

                data = response.json()["data"]
                balance = data["total_credits"] - data["total_usage"]

                results.append(f"**Open Router balance:** ${balance:.2f}")

            except requests.exceptions.RequestException as e:
                if hasattr(e, "response") and e.response is not None:
                    error_msg = f"Error retrieving balance from Open Router: {e.response.status_code} - {e.response.text}"
                else:
                    error_msg = f"Error retrieving balance from Open Router: {str(e)}"
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"An unexpected error occurred while fetching balance for Open Router: {str(e)}"
                errors.append(error_msg)

        # Combine results and errors
        results_str = "\n\n".join(results)  # Add blank line between provider results
        errors_str = "\n".join(errors)

        final_message = results_str
        if errors_str:
            final_message += (
                "\n\n" if results_str else ""
            ) + errors_str  # Add blank line between results and errors

        if not results and not errors:
            return "Error: No balance providers configured."

        return final_message

    def get_table_schema(self):
        """Get the usage_costs table schema based on database type"""
        is_sqlite = "sqlite" in engine.url.drivername

        if is_sqlite:
            query = "PRAGMA table_info(usage_costs);"
        else:
            query = """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'usage_costs'
                ORDER BY ordinal_position;
            """

        with get_db() as db:
            result = db.execute(text(query))
            rows = result.fetchall()

        if is_sqlite:
            schema = "\n".join([f"- {row.name}: {row.type}" for row in rows])
        else:
            schema = "\n".join(
                [f"- {row.column_name}: {row.data_type}" for row in rows]
            )

        return schema

    def get_user_api_key(self, user_id):
        """Get the user's API key from the database"""
        is_sqlite = "sqlite" in engine.url.drivername

        if is_sqlite:
            query = "SELECT api_key FROM user WHERE id = :user_id;"
        else:
            # For PostgreSQL, explicitly use the public schema
            query = "SELECT api_key FROM public.user WHERE id = :user_id;"

        with get_db() as db:
            result = db.execute(text(query), {"user_id": user_id})
            row = result.fetchone()

            return row.api_key if row else None

    async def handle_ask_command(self, __user__, body, question):
        """Handle natural language questions about usage data"""
        # Get user's API key
        api_key = self.get_user_api_key(__user__["id"])
        if not api_key:
            return (
                "Error: You must have an API key generated to use this feature.\n"
                "Please go to Settings -> Account to generate an API key."
            )

        # Get database type and schema
        is_sqlite = "sqlite" in engine.url.drivername
        db_type = "SQLite" if is_sqlite else "PostgreSQL"
        schema = self.get_table_schema()

        # Construct the prompt
        prompt = (
            get_messages_content(body["messages"])
            + f"""^^ THIS WAS PRIOR CONVERSATION CONTEXT^^
                        
                        NOW, you are a SQL query generator. Generate a SQL query for the following question:

                Question: {question}

                Database Type: {db_type}
                Table: usage_costs
                Schema:
                {schema}

                Note: the Task column is NULL for the regular chat requests; Task can be "title_generation", "tags_generation", "query_generation", "autocomplete_generation" made by the UI tool that accompany chats.
                Make a reasonable assumption about the users intention if they want information only from main chat completion requests (Task is NULL) or to include task usage. 
                For costs summarization, typically all tasks can be included. If a breakdown by model is requested, probably only main chat completions should be included. 
                For counting usage/requests, only main chat completions should be included. If unsure, consider building the report to separately highlight both numbers.

                The query must start with SELECT and end with a semicolon. Generate only the SQL query, nothing else. Do not use WITH or CTE clauses."""
        )

        try:
            # Create AsyncOpenAI client
            client = AsyncOpenAI(base_url="http://localhost:8080/api", api_key=api_key)

            # Get SQL query from LLM using async call
            completion = await client.chat.completions.create(
                model=self.valves.SQL_ASSISTANT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a SQL expert assistant."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )

            # Extract and clean query from response
            sql_query = completion.choices[0].message.content.strip()
            sql_query = re.sub(r"^```sql\s*|\s*```$", "", sql_query, flags=re.MULTILINE)
            sql_query = sql_query.strip()

            # Validate query
            if not re.match(r"^SELECT", sql_query, re.IGNORECASE):
                return "Error: executable query not obtained.\n" + sql_query

            if not sql_query.rstrip().endswith(";"):
                sql_query += ";"

            # Execute the query
            result = self.run_sql_command(sql_query)

            # Format the response
            response = "Generated SQL Query:\n```sql\n"
            response += sql_query + "\n```\n\n"
            response += result

            return response

        except Exception as e:
            _, _, tb = sys.exc_info()
            error_msg = f"Error on line {tb.tb_lineno}: {str(e)}"
            self.logger.error(f"Error processing ask command: {error_msg}")
            return f"Error processing question: {str(e)}"
