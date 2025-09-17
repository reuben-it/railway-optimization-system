import { StyleSheet, Text, View } from "react-native";

export default function OperatorHome() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>📊 Station Operator Dashboard</Text>
      <Text>Here you can manage conflicts, delays, and platform assignments.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center", alignItems: "center" },
  title: { fontSize: 22, fontWeight: "bold", marginBottom: 10 },
});
