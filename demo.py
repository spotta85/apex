"""End-to-end demo of the basic cache + async prefetch pipeline."""

from apex.demo_modes import make_basic_runtime

bundle = make_basic_runtime()

print("First run:")
bundle.runtime.run()

print("\nSecond run:")
final = bundle.runtime.run()

print("\nFinal hidden state:")
print(final)

bundle.prefetcher.shutdown()
