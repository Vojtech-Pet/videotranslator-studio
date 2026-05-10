from __future__ import annotations


EVAL_SET = [
    {
        "name": "sql_null_basic",
        "input": "This is what a null means in SQL.",
        "must_contain": ["NULL", "SQL"],
    },
    {
        "name": "isnull_basic",
        "input": "We can use the ISNULL function.",
        "must_contain": ["ISNULL"],
    },
    {
        "name": "coalesce_basic",
        "input": "The second function is COALESCE.",
        "must_contain": ["COALESCE"],
    },
    {
        "name": "is_not_null_basic",
        "input": "We can use IS NOT NULL.",
        "must_contain": ["IS NOT NULL"],
    },
]
