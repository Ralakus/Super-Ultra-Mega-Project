"""Traversal graph."""

from pydantic import BaseModel


class Path(BaseModel):
    """Path type."""

    source: str
    target: str


class Graph(BaseModel):
    """Traversal Graph."""

    name: str
    paths: list[Path]

    def graphviz_dot_dump(self) -> str:
        """Dump graph into graphviz dot string.

        Returns:
            str: graphviz dot
        """
        relationships: str = "\n".join(f"\t\"{path.source}\" -> \"{path.target}\";" for path in self.paths)
        return f"digraph D {{\n{relationships}\n}}"
