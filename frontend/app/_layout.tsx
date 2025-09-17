import { Stack } from "expo-router";

export default function RootLayout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      {/* All screens under /app will now have no header */}
    </Stack>
  );
}