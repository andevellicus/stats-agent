import socket
import sys
import io
import os

# A special token to signal the end of a message.
EOM_TOKEN = "<|EOM|>"

# This is now our session manager.
# It will map a session_id to its own private state dictionary.
sessions = {}

def execute_code(session_id, code):
    """Executes code within a specific session's state."""
    # Get the private state for this session.
    # If the session doesn't exist yet, create an empty state for it.
    if session_id not in sessions:
        sessions[session_id] = {}
    
    session_state = sessions[session_id]

    # Create and change to the session-specific workspace directory
    workspace_dir = os.path.join('/app/workspaces', session_id)
    os.makedirs(workspace_dir, exist_ok=True)
    
    original_dir = os.getcwd()
    os.chdir(workspace_dir)


    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        # We execute the code using the session_state and capture any printed output.
        exec(code, session_state)
        output = redirected_output.getvalue()
        return output
    except Exception as e:
        # Return a more descriptive error message
        return f"Error: {type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout
        os.chdir(original_dir)


def main():
    """Listens for connections and executes code in sandboxed sessions."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 9999))
    server_socket.listen(5)
    print("Python session server is listening on port 9999...")

    while True:
        conn, addr = server_socket.accept()
        print(f"Accepted connection from {addr}")
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                
                # We expect the message to be in the format: "session_id|code"
                message = data.decode('utf-8')
                parts = message.split('|', 1)
                if len(parts) != 2:
                    conn.sendall(b"Error: Invalid message format. Expected 'session_id|code'.")
                    continue

                session_id, code = parts

                # --- Log the execution ---
                print(f"=== Session {session_id} ===")
                print("Executing Python code:")

                # Print code with line numbers
                lines = code.split('\n')
                for i, line in enumerate(lines, 1):
                    print(f"{i:3d} | {line}")

                print("-" * 30)

                result = execute_code(session_id, code)

                # If the result is empty, send a confirmation message to prevent hanging.
                if not result.strip():
                    result = "Success: Code executed with no output."

                print(f"Result: {result}")
                print("=" * 30)

                # Append the EOM token to the result before sending.
                message_to_send = result + EOM_TOKEN
                conn.sendall(message_to_send.encode('utf-8'))
        except socket.error as e:
            print(f"Socket error: {e}")
        finally:
            print(f"Closing connection from {addr}")
            conn.close()

if __name__ == "__main__":
    main()