import socket
import sys
import io

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

    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        # Crucially, we execute the code using the *session_state*
        exec(code, session_state)
        output = redirected_output.getvalue()
        try:
            last_line = code.strip().split('\n')[-1]
            result = eval(last_line, session_state)
            if result is not None:
                output += str(result)
        except:
            pass
        return output
    except Exception as e:
        # Return a more descriptive error message
        return f"Error: {type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout

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
                print(f"Result: {result}")
                print("=" * 30)
                conn.sendall(result.encode('utf-8'))
        except socket.error as e:
            print(f"Socket error: {e}")
        finally:
            print(f"Closing connection from {addr}")
            conn.close()

if __name__ == "__main__":
    main()