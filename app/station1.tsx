import { useState } from "react";
import { ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from "react-native";

export default function Dashboard() {
  const [input, setInput] = useState("");
  const [justification, setJustification] = useState("");

  return (
    <View style={styles.container}>
      {/* Top Input Area */}
      
      {/* Main Content - 3 Panels */}
      <View style={styles.main}>
        {/* Left Panel - Parsed JSON */}
        <ScrollView style={styles.panel}>
          <Text style={styles.title}>Parsed JSON</Text>
          <Text style={styles.code}>
{`{
  "train": "1234",
  "status": "delayed",
  "cause": "maintenance"
}`}
          </Text>
        </ScrollView>

        {/* Center Panel - ML Outputs */}
        <ScrollView style={styles.panel}>
          <Text style={styles.title}>ML Outputs</Text>
          <Text>Conflict Likelihood: 0.72</Text>
          <Text>Propagated Delay: 14 min</Text>
        </ScrollView>

        {/* Right Panel - LLM Recommendations */}
        <ScrollView style={styles.panel}>
          <Text style={styles.title}>LLM Recommendations</Text>
          <Text>Divert train 1234 to Platform 2.</Text>
          <Text style={styles.fallback}>Fallback rule: use nearest available platform.</Text>
        </ScrollView>
      </View>

      {/* Footer */}
      <View style={styles.footer}>
        <TouchableOpacity style={[styles.button, { backgroundColor: "#16a34a" }]}>
          <Text style={styles.buttonText}>Accept</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.button, { backgroundColor: "#facc15" }]}>
          <Text style={styles.buttonText}>Override</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.button, { backgroundColor: "#dc2626" }]}>
          <Text style={styles.buttonText}>Reject</Text>
        </TouchableOpacity>
        <TextInput
          value={justification}
          onChangeText={setJustification}
          placeholder="Enter justification..."
          style={styles.justificationBox}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f9fafb" },
  top: { padding: 10, borderBottomWidth: 1, borderColor: "#ddd" },
  inputBox: {
    backgroundColor: "white",
    borderWidth: 1,
    borderColor: "#ddd",
    borderRadius: 8,
    padding: 10,
    textAlignVertical: "top",
    minHeight: 60,
  },
  main: { flex: 1, flexDirection: "row" },
  panel: {
    flex: 1,
    padding: 10,
    borderRightWidth: 1,
    borderColor: "#ddd",
    backgroundColor: "white",
  },
  title: { fontWeight: "bold", marginBottom: 8 },
  code: {
    fontFamily: "monospace",
    backgroundColor: "#f3f4f6",
    padding: 8,
    borderRadius: 6,
  },
  fallback: { marginTop: 8, color: "#6b7280", fontSize: 12 },
  footer: {
    flexDirection: "row",
    alignItems: "center",
    padding: 10,
    borderTopWidth: 1,
    borderColor: "#ddd",
    backgroundColor: "#f9fafb",
  },
  button: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  buttonText: { color: "white", fontWeight: "bold" },
  justificationBox: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#ddd",
    borderRadius: 6,
    padding: 8,
    backgroundColor: "white",
  },
});
