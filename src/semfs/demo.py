"""Generate a small, reproducible demo directory. Never overwrite existing files."""

import argparse
import shutil
from pathlib import Path

DOCUMENTS = {
    "notes/operating_systems.md": """# CPU scheduling
An operating system chooses the next process to run on the CPU.
Round robin assigns time slices. Shortest job first minimizes average waiting time.
Priority scheduling can cause starvation; aging increases waiting processes' priorities.

# Deadlocks
Deadlock occurs when processes wait forever for resources held by each other.
The four conditions are mutual exclusion, hold and wait, no preemption, circular wait.
Prevention breaks one of these conditions. Banker's algorithm avoids unsafe states.
""",
    "notes/synchronization.md": """# Mutexes and semaphores
Concurrent threads can corrupt shared data without synchronization.
A mutex protects a critical section. A counting semaphore manages multiple resources.
Always acquire locks in the same order to prevent circular wait and deadlocks.
Condition variables let threads wait until a shared state changes.
""",
    "notes/concurrency_summary.txt": """Threads coordinate access to shared memory with locks.
A mutual exclusion lock allows one thread into a critical section at a time.
Semaphores limit the number of tasks accessing a resource.
Consistent lock ordering helps avoid deadlock between concurrent processes.
""",
    "code/graphs/dijkstra.py": '''"""Shortest paths from one vertex using Dijkstra's algorithm."""
import heapq


def dijkstra(graph, start):
    distances = {start: 0}
    queue = [(0, start)]
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances[vertex]:
            continue
        for neighbor, weight in graph.get(vertex, []):
            candidate = distance + weight
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances
''',
    "code/hello.c": """#include <stdio.h>
int main(void) {
    printf("Hello from the operating systems lab!\\n");
    return 0;
}
""",
    "personal/gardening.txt": """Tomatoes need sunlight, well-drained soil and regular watering.
Compost improves soil fertility. Plant basil near tomatoes to attract pollinators.
""",
    "personal/recipe.md": """# Tomato soup
Roast ripe tomatoes with garlic and olive oil. Blend with vegetable stock.
Simmer for twenty minutes and finish with fresh basil.
""",
}


def generate(destination: Path) -> Path:
    destination = destination.expanduser().absolute()
    if destination.exists():
        raise FileExistsError(f"Choose a new directory; {destination} already exists")
    destination.mkdir(parents=True)
    for relative, text in DOCUMENTS.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    copies = destination / "copies"
    copies.mkdir()
    shutil.copy2(destination / "notes/operating_systems.md", copies / "os_backup.md")
    shutil.copy2(destination / "notes/operating_systems.md", copies / "os_second_copy.md")
    shutil.copy2(destination / "code/graphs/dijkstra.py", copies / "dijkstra_backup.py")

    import pymupdf
    from PIL import Image, ImageDraw

    with pymupdf.open() as document:
        for title, body in [
            (
                "CPU scheduling",
                "Round robin uses a fixed quantum. Priority scheduling selects "
                "the highest priority runnable process. Aging prevents starvation.",
            ),
            (
                "Deadlock prevention",
                "Deadlock requires mutual exclusion, hold and wait, no "
                "preemption, and circular wait. Resource ordering breaks circular wait. "
                "Semaphores and mutexes coordinate concurrent threads.",
            ),
        ]:
            page = document.new_page()
            page.insert_text((60, 75), title, fontsize=22)
            page.insert_textbox((60, 110, 520, 700), body, fontsize=14)
        document.save(destination / "notes/os_handbook.pdf")

    images = destination / "images"
    images.mkdir()
    for color in ("red", "blue"):
        image = Image.new("RGB", (640, 360), "#edf5f7")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 270, 640, 360), fill="#53616b")
        draw.rounded_rectangle((100, 165, 545, 280), radius=25, fill=color)
        draw.polygon([(210, 165), (265, 85), (400, 85), (470, 165)], fill=color)
        draw.polygon([(240, 155), (280, 100), (330, 100), (330, 155)], fill="#b4e5f6")
        draw.polygon([(345, 155), (345, 100), (390, 100), (440, 155)], fill="#b4e5f6")
        for x in (205, 455):
            draw.ellipse((x - 38, 240, x + 38, 316), fill="#17212a")
            draw.ellipse((x - 18, 260, x + 18, 296), fill="#b9c4ce")
        # Neutral filenames ensure that image search must use visual content.
        image.save(images / ("scene_01.png" if color == "red" else "scene_02.png"))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the SemFS demonstration dataset")
    parser.add_argument("folder", nargs="?", type=Path, default=Path("demo-files"))
    args = parser.parse_args()
    try:
        print(generate(args.folder))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
