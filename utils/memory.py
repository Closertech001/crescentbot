# Simple memory tracker for Streamlit session
def store_last_query_info(state, query_info):
    state["last_query_info"] = query_info

def get_last_query_info(state):
    return state.get("last_query_info", None)
