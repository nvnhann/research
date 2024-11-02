package com.udacity.webcrawler.profiler;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.util.Objects;

/**
 * A method interceptor that checks whether {@link Method}s are annotated with the {@link Profiled}
 * annotation. If they are, the method interceptor records how long the method invocation took.
 */
final class ProfilingMethodInterceptor implements InvocationHandler {

  private final Clock clock;
  private final Object delegate;
  private final ProfilingState state;

  ProfilingMethodInterceptor(Clock clock, Object delegate, ProfilingState state) {
    this.clock = Objects.requireNonNull(clock);
    this.delegate = delegate;
    this.state = state;
  }

  @Override
  public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
    if (method.isAnnotationPresent(Profiled.class)) {
      // If the method is annotated with @Profiled
      Instant start = clock.instant();
      try {
        // Execute the method and retrieve the result
        return method.invoke(delegate, args);
      } catch (InvocationTargetException e) {
        // Handle any exceptions thrown during execution
        throw e.getTargetException();
      } catch (IllegalAccessException e) {
        // Throw an exception if there's an access issue
        throw new RuntimeException(e);
      } finally {
        // Measure the execution time and record it to ProfilingState
        state.record(delegate.getClass(), method, Duration.between(start, clock.instant()));
      }
    }
    // If the method is not annotated with @Profiled, simply execute the method and return the
    // result
    return method.invoke(delegate, args);
  }

}
