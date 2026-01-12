import { useState, useRef, useEffect } from "react";
import { Send, Loader } from "lucide-react";

export default function UniversityChatbot() {
    const [messages, setMessages] = useState([
        {
            id: 1,
            type: "bot",
            text: "Hello! I'm here to help you with questions about fees, programs, admissions, and more. How can I assist you today?",
            sources: [],
        },
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const sendMessage = async () => {
        if (!input.trim()) return;

        const userMessage = {
            id: messages.length + 1,
            type: "user",
            text: input,
            sources: [],
        };

        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        setLoading(true);

        try {
            const response = await fetch("http://localhost:8000/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ query: input }),
            });

            const data = await response.json();

            const botMessage = {
                id: messages.length + 2,
                type: "bot",
                text:
                    data.answer ||
                    "I apologize, but I couldn't find information about that.",
                sources: data.sources || [],
            };

            setMessages((prev) => [...prev, botMessage]);
        } catch {
            const errorMessage = {
                id: messages.length + 2,
                type: "bot",
                text: "Sorry, I encountered an error connecting to the server. Please try again later.",
                sources: [],
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    return (
        <div className="min-h-screen min-w-screen bg-gray-50 flex flex-col">
            {/* Header */}
            <header className="bg-white border-b border-gray-200">
                <div className="w-full px-6 py-4 flex items-center gap-4">
                    <div className="w-12 h-12 bg-linear-to-br from-blue-400 to-blue-600 rounded-lg flex items-center justify-center">
                        <span className="text-white font-bold text-lg">U</span>
                    </div>
                    <div>
                        <h1 className="text-xl font-bold text-gray-900">
                            Student Help Center
                        </h1>
                        <p className="text-sm text-gray-600">
                            facultie de science de bizerte
                        </p>
                    </div>
                </div>
            </header>

            {/* Main Chat Area */}
            <main className="flex-1 overflow-auto">
                <div className="w-full px-6 py-8">
                    <div className="space-y-6">
                        {messages.map((message) => (
                            <div
                                key={message.id}
                                className={`flex ${
                                    message.type === "user"
                                        ? "justify-end"
                                        : "justify-start"
                                }`}
                            >
                                <div
                                    className={`max-w-2xl ${
                                        message.type === "user"
                                            ? "bg-blue-600 text-white rounded-2xl rounded-tr-none"
                                            : "bg-gray-100 text-gray-900 rounded-2xl rounded-tl-none"
                                    } px-6 py-4 shadow-sm`}
                                >
                                    <p className="text-base leading-relaxed whitespace-pre-wrap">
                                        {message.text}
                                    </p>

                                    {/* Sources */}
                                    {message.sources &&
                                        message.sources.length > 0 && (
                                            <div className="mt-4 pt-4 border-t border-gray-300">
                                                <p
                                                    className={`text-xs font-semibold mb-2 ${
                                                        message.type === "user"
                                                            ? "text-blue-100"
                                                            : "text-gray-600"
                                                    }`}
                                                >
                                                    SOURCES:
                                                </p>
                                                <div className="space-y-2">
                                                    {message.sources.map(
                                                        (source, idx) => (
                                                            <div
                                                                key={idx}
                                                                className={`text-xs p-2 rounded ${
                                                                    message.type ===
                                                                    "user"
                                                                        ? "bg-blue-500 text-blue-50"
                                                                        : "bg-blue-50 text-blue-800"
                                                                }`}
                                                            >
                                                                📄 {source}
                                                            </div>
                                                        )
                                                    )}
                                                </div>
                                            </div>
                                        )}
                                </div>
                            </div>
                        ))}

                        {loading && (
                            <div className="flex justify-start">
                                <div className="bg-gray-100 text-gray-900 rounded-2xl rounded-tl-none px-6 py-4 shadow-sm">
                                    <div className="flex items-center gap-2">
                                        <Loader className="w-5 h-5 animate-spin text-blue-600" />
                                        <span className="text-base">
                                            Searching for information...
                                        </span>
                                    </div>
                                </div>
                            </div>
                        )}

                        <div ref={messagesEndRef} />
                    </div>
                </div>
            </main>

            {/* Input Area */}
            <footer className="bg-white border-t border-gray-200 sticky bottom-0">
                <div className="w-full px-6 py-6">
                    <div className="flex gap-3">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="Ask about fees, programs, admissions, scholarships..."
                            style={{color:"black"}}
                            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-100 text-base"
                            disabled={loading}
                        />
                        <button
                            onClick={sendMessage}
                            disabled={loading || !input.trim()}
                            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white px-6 py-3 rounded-lg font-medium transition-colors flex items-center gap-2"
                        >
                            <Send className="w-5 h-5" />
                            <span className="hidden sm:inline">Send</span>
                        </button>
                    </div>
                    <p className="text-xs text-gray-500 mt-3">
                        💡 Tip: You can ask about fees, academic programs,
                        admissions, deadlines, and more.
                    </p>
                </div>
            </footer>
        </div>
    );
}
