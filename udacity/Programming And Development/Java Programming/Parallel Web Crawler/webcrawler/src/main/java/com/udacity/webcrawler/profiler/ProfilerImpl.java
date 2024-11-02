package com.udacity.webcrawler.profiler;

import javax.inject.Inject;
import java.io.IOException;
import java.io.Writer;
import java.lang.reflect.Proxy;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Clock;
import java.time.ZonedDateTime;
import java.util.Arrays;
import java.util.Objects;

import static java.time.format.DateTimeFormatter.RFC_1123_DATE_TIME;

/**
 * Concrete implementation of the {@link Profiler}.
 */
final class ProfilerImpl implements Profiler {

  private final Clock clock;
  private final ProfilingState state = new ProfilingState();
  private final ZonedDateTime startTime;

  @Inject
  ProfilerImpl(Clock clock) {
    this.clock = Objects.requireNonNull(clock);
    this.startTime = ZonedDateTime.now(clock);
  }

  @Override
  @SuppressWarnings("unchecked")
  public <T> T wrap(Class<T> klass, T delegate) {
    Objects.requireNonNull(klass);

    // Use a dynamic proxy (java.lang.reflect.Proxy) to "wrap" the delegate in a
    // ProfilingMethodInterceptor and return a dynamic proxy from this method.
    // See https://docs.oracle.com/javase/10/docs/api/java/lang/reflect/Proxy.html.
    boolean hasProfiledMethod = Arrays.stream(klass.getDeclaredMethods()).anyMatch(method -> Arrays
        .stream(method.getAnnotations()).anyMatch(annotation -> annotation instanceof Profiled));

    if (!hasProfiledMethod) {
      throw new IllegalArgumentException("No profiled methods found in the provided class.");
    }

    return (T) Proxy.newProxyInstance(delegate.getClass().getClassLoader(), new Class<?>[] {klass},
        new ProfilingMethodInterceptor(clock, delegate, state));
  }

  @Override
  public void writeData(Path path) {
    // Write the ProfilingState data to the given file path. If a file already exists at that
    // path, the new data should be appended to the existing file.
    try (Writer wr = Files.newBufferedWriter(path, StandardCharsets.UTF_8, StandardOpenOption.CREATE,
          StandardOpenOption.WRITE, StandardOpenOption.APPEND) ) {
      writeData(wr);
      wr.flush();
    } catch (IOException e) {
      // Auto-generated catch block
      e.printStackTrace();
    }

  }

  @Override
  public void writeData(Writer writer) {
    try {
      writer.write("Run at " + RFC_1123_DATE_TIME.format(startTime));
      writer.write(System.lineSeparator());
      state.write(writer);
      writer.write(System.lineSeparator());
    } catch (IOException e) {
      // Auto-generated catch block
      e.printStackTrace();
    }

  }
}
