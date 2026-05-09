"""Workflow implementations for contract-question-agent."""

import warnings

warnings.filterwarnings("ignore", message=r".*is experimental.*")

from contract_question_agent.workflows.workflow import run_linear_workflow

__all__ = ["run_linear_workflow"]
